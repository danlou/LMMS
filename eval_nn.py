import os
import logging
import argparse
from time import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
from nltk.corpus import wordnet as wn

from bert_as_service import bert_embed
from vectorspace import SensesVSM
from vectorspace import get_sk_pos


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def load_wsd_fw_set(wsd_fw_set_path):
    """Parse XML of split set and return list of instances (dict)."""
    eval_instances = []
    tree = ET.parse(wsd_fw_set_path)
    for text in tree.getroot():
        for sent_idx, sentence in enumerate(text):
            inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
            for e in sentence:
                inst['tokens_mw'].append(e.text)
                inst['lemmas'].append(e.get('lemma'))
                inst['senses'].append(e.get('id'))
                inst['pos'].append(e.get('pos'))

            inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

            # handling multi-word expressions, mapping allows matching tokens with mw features
            idx_map_abs = []
            idx_map_rel = [(i, list(range(len(t.split()))))
                            for i, t in enumerate(inst['tokens_mw'])]
            token_counter = 0
            for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                idx_tokens = [i+token_counter for i in idx_tokens]
                token_counter += len(idx_tokens)
                idx_map_abs.append([idx_group, idx_tokens])

            inst['tokenized_sentence'] = ' '.join(inst['tokens'])
            inst['idx_map_abs'] = idx_map_abs
            inst['idx'] = sent_idx

            eval_instances.append(inst)

    return eval_instances


@lru_cache()
def wn_sensekey2synset(sensekey):
    """Convert sensekey to synset."""
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


def get_id2sks(wsd_eval_keys):
    """Maps ids of split set to sensekeys, just for in-code evaluation."""
    id2sks = {}
    with open(wsd_eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2sks[id_] = keys
    return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
    """Runs the official java-based scorer of the WSD Evaluation Framework."""
    cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
                                          '%s/%s.gold.key.txt' % (test_set, test_set),
                                          '../../../../' + results_path)
    print(cmd)
    os.system(cmd)


def chunks(l, n):
    """Yield successive n-sized chunks from given list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def str_scores(scores, n=3, r=5):
    """Convert scores list to a more readable string."""
    return str([(l, round(s, r)) for l, s in scores[:n]])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
    parser.add_argument('-ft_path', help='Path to fastText vectors', required=False,
                        default='external/fasttext/crawl-300d-2M-subword.bin')
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-test_set', default='ALL', help='Name of test set', required=False,
                        choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
    parser.add_argument('-ignore_lemma', dest='use_lemma', action='store_false', help='Ignore lemma features', required=False)
    parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
    parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
    parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
    parser.add_argument('-quiet', dest='debug', action='store_false', help='Less verbose (debug=False)', required=False)
    parser.set_defaults(use_lemma=True)
    parser.set_defaults(use_pos=True)
    parser.set_defaults(debug=True)
    args = parser.parse_args()

    # logging.info('Running with Parameters:')
    # logging.info('Sense Vectors Path (sv_path) - %s' % args.sv_path)
    # logging.info('fastText Vectors Path (ft_path) - %s' % args.ft_path)
    # logging.info('WSD Eval Framework Path (wsd_path) - %s' % args.wsd_fw_path)
    # logging.info('Evaluation Test Set (test_set) - %s' % args.test_set)
    # logging.info('Batch Size (batch_size) - %d' % args.batch_size)
    # logging.info('WordPiece Merge Strategy (merge_strategy) - %s' % args.merge_strategy)
    # logging.info('Use Lemma (ignore_lemma) - %s' % args.use_lemma)  #TO-DO: fix naming
    # logging.info('Use Part-of-Speech (ignore_pos) - %s' % args.use_pos)  #TO-DO: fix namings
    # logging.info('Number of Neighbors Accepted (k) - %d' % args.k)
    # logging.info('Similarity Threshold (t) - %f' % args.thresh)

    """
    Load sense embeddings for evaluation.
    Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
    Load fastText static embeddings if required.
    """
    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.sv_path, normalize=True)
    logging.info('Loaded SensesVSM')

    ft_model = None
    if senses_vsm.ndims in [300+1024, 300+1024+1024]:
        logging.info('SensesVSM requires fastText')
        if args.ft_path != '':
            logging.info('Loading pretrained fastText ...')
            import fastText  # importing here so that fastText is an optional requirement
            ft_model = fastText.load_model(args.ft_path)
            logging.info('Loaded pretrained fastText')
        else:
            logging.critical('fastText model is undefined and expected by SensesVSM.')
            raise Exception('Input Failure')

    """
    Initialize various counters for calculating supplementary metrics.
    """
    n_instances, n_correct, n_unk_lemmas = 0, 0, 0
    correct_idxs = []
    num_options = []
    failed_by_pos = defaultdict(list)

    pos_confusion = {}
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}

    """
    Load evaluation instances and gold labels.
    Gold labels (sensekeys) only used for reporting accuracy during evaluation.
    """
    wsd_fw_set_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
    wsd_fw_gold_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)
    id2senses = get_id2sks(wsd_fw_gold_path)
    eval_instances = load_wsd_fw_set(wsd_fw_set_path)
    
    """
    Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
    File with predictions is processed by the official scorer after iterating over all instances.
    """
    results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.merge_strategy)
    with open(results_path, 'w') as results_f:
        for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):
            batch_sents = [sent_info['tokenized_sentence'] for sent_info in batch]

            # process contextual embeddings in sentences batches of size args.batch_size
            batch_bert = bert_embed(batch_sents, merge_strategy=args.merge_strategy)

            for sent_info, sent_bert in zip(batch, batch_bert):
                idx_map_abs = sent_info['idx_map_abs']

                for mw_idx, tok_idxs in idx_map_abs:
                    curr_sense = sent_info['senses'][mw_idx]

                    if curr_sense is None:
                        continue

                    curr_lemma = sent_info['lemmas'][mw_idx]

                    if args.use_lemma and curr_lemma not in senses_vsm.known_lemmas:
                        continue  # skips hurt performance in official scorer

                    curr_postag = sent_info['pos'][mw_idx]
                    curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
                    curr_vector = np.array([sent_bert[i][1] for i in tok_idxs]).mean(axis=0)
                    curr_vector = curr_vector / np.linalg.norm(curr_vector)

                    """
                    Fetch (or compose) static embedding if it's expected.
                    Uses lemma by default unless specified otherwise by CLI parameter.
                    This lemma is a gold feature of the evaluation datasets.
                    """
                    static_vector = None
                    if ft_model is not None:
                        if args.use_lemma:
                            static_vector = ft_model.get_word_vector(curr_lemma)
                        else:
                            static_vector = ft_model.get_word_vector('_'.join(curr_tokens))
                        static_vector = static_vector / np.linalg.norm(static_vector)

                    """
                    Compose test-time embedding for matching with sense embeddings in SensesVSM.
                    Test-time embedding corresponds to stack of contextual and (possibly) static embeddings.
                    Stacking composition performed according to dimensionality of sense embeddings.
                    """
                    if senses_vsm.ndims == 1024:
                        curr_vector = curr_vector

                    # duplicating contextual feature for cos similarity against features from
                    # sense annotations and glosses that belong to the same NLM
                    elif senses_vsm.ndims == 1024+1024:
                        curr_vector = np.hstack((curr_vector, curr_vector))

                    elif senses_vsm.ndims == 300+1024 and static_vector is not None:
                        curr_vector = np.hstack((static_vector, curr_vector))

                    elif senses_vsm.ndims == 300+1024+1024 and static_vector is not None:
                        curr_vector = np.hstack((static_vector, curr_vector, curr_vector))

                    curr_vector = curr_vector / np.linalg.norm(curr_vector)

                    """
                    Matches test-time embedding against sense embeddings in SensesVSM.
                    use_lemma and use_pos flags condition filtering of candidate senses.
                    Matching is actually cosine similarity (most similar), or 1-NN.
                    """
                    matches = []
                    if args.use_lemma and curr_lemma not in senses_vsm.known_lemmas:
                        n_unk_lemmas += 1

                    elif args.use_lemma and args.use_pos:  # the usual for WSD
                        matches = senses_vsm.match_senses(curr_vector, curr_lemma, curr_postag, topn=None)

                    elif args.use_lemma:
                        matches = senses_vsm.match_senses(curr_vector, curr_lemma, None, topn=None)

                    elif args.use_pos:
                        matches = senses_vsm.match_senses(curr_vector, None, curr_postag, topn=None)

                    else:  # corresponds to Uninformed Sense Matching (USM)
                        matches = senses_vsm.match_senses(curr_vector, None, None, topn=None)

                    num_options.append(len(matches))

                    # predictions can be further filtered by similarity threshold or number of accepted neighbors
                    # if specified in CLI parameters
                    preds = [sk for sk, sim in matches if sim > args.thresh][:args.k]

                    if len(preds) > 0:
                        results_f.write('%s %s\n' % (curr_sense, preds[0]))

                    """
                    Processing additional performance metrics.
                    """

                    # check if our prediction(s) was correct, register POS of mistakes
                    n_instances += 1
                    wsd_correct = False
                    gold_sensekeys = id2senses[curr_sense]
                    if len(set(preds).intersection(set(gold_sensekeys))) > 0:
                        n_correct += 1
                        wsd_correct = True
                    else:
                        if len(preds) > 0:
                            failed_by_pos[curr_postag].append((preds[0], gold_sensekeys))
                        else:
                            failed_by_pos[curr_postag].append((None, gold_sensekeys))

                    # register if our prediction belonged to a different POS than gold
                    if len(preds) > 0:
                        pred_sk_pos = get_sk_pos(preds[0])
                        gold_sk_pos = get_sk_pos(gold_sensekeys[0])
                        pos_confusion[gold_sk_pos][pred_sk_pos] += 1

                    # register how far the correct prediction was from the top of our matches
                    correct_idx = None
                    for idx, (matched_sensekey, matched_score) in enumerate(matches):
                        if matched_sensekey in gold_sensekeys:
                            correct_idx = idx
                            correct_idxs.append(idx)
                            break

                    if args.debug:
                        acc = n_correct / n_instances
                        logging.debug('ACC: %.3f (%d %d/%d)' % (
                            acc, n_instances, sent_info['idx'], len(eval_instances)))

                        # # Additional debugging, very verbose and unnecessary most of the time
                        # gold_synsets = [wn_sensekey2synset(k).name() for k in gold_sensekeys]
                        # logging.debug('')
                        # logging.debug('instance: %d' % instances)
                        # logging.debug('sentence: %s' % sent_info['tokenized_sentence'])
                        # logging.debug('tokens: %s' % curr_tokens)
                        # logging.debug('postag: %s' % curr_postag)
                        # logging.debug('lemma: %s' % curr_lemma)
                        # logging.debug('matches: %s %d' % (str_scores(matches, 5), len(matches)))
                        # logging.debug('pred: %s' % preds)
                        # logging.debug('gold: %s %s' % (gold_sensekeys, gold_synsets))
                        # # logging.debug('correct idx: %d' % correct_idx)
                        # if wsd_correct:
                        #     logging.debug('CORRECT')
                        # else:
                        #     logging.debug('WRONG')

    if args.debug:
        """
        Summary of supplementary performance metrics.
        """
        logging.info('Supplementary Metrics:')
        logging.info('Avg. correct idx: %.6f' % np.mean(np.array(correct_idxs)))
        logging.info('Avg. correct idx (failed): %.6f' % np.mean(np.array([i for i in correct_idxs if i > 0])))
        logging.info('Avg. num options: %.6f' % np.mean(num_options))
        logging.info('Num. unknown lemmas: %d' % n_unk_lemmas)

        logging.info('POS Failures:')
        for pos, fails in failed_by_pos.items():
            logging.info('%s fails: %d' % (pos, len(fails)))

        logging.info('POS Confusion:')
        for pos in pos_confusion:
            logging.info('%s - %s' % (pos, str(pos_confusion[pos])))

    logging.info('Running official scorer ...')
    run_scorer(args.wsd_fw_path, args.test_set, results_path)
