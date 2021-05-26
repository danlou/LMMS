"""
Example usage (from root folder):
python evaluation/eval_wsd.py -nlm_id albert-xxlarge-v2 -lmms_path data/vectors/albert-xxlarge-v2/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt -weights_path data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt -layer_op ws
"""

import subprocess
import logging
import argparse
import xml.etree.ElementTree as ET

import numpy as np
from nltk.corpus import wordnet as wn

import sys, os  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder

from vectorspace import SensesVSM


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')



def load_wsd_fw_set(wsd_fw_set_path):
    """Parse XML of split set and return list of instances (dict)."""

    def convert_pos(postag):
        short2long_map = {'VB': 'VERB', 'NN': 'NOUN', 'JJ': 'ADJ', 'RB': 'ADV'}
        if postag[:2] in short2long_map:
            return short2long_map[postag[:2]]
        else:
            return postag

    eval_instances = []
    tree = ET.parse(wsd_fw_set_path)
    for text in tree.getroot():
        for sent_idx, sentence in enumerate(text):
            inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
            for e in sentence:
                inst['tokens_mw'].append(e.text)
                inst['lemmas'].append(e.get('lemma'))
                inst['senses'].append(e.get('id'))
                inst['pos'].append(convert_pos(e.get('pos')))

            inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

            if set(inst['senses']) == {None}:
                continue

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


def load_ground_truth(eval_keys, sense_level='sensekey'):
    """ Maps ids of split set to sensekeys, just for in-code evaluation. """
    id2sks = {}
    with open(eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            if sense_level == 'synset':
                keys = [map_sk2syn[k] for k in keys]
            id2sks[id_] = keys
    return id2sks


def run_scorer(eval_fw_path, test_set, results_path):
    """ Runs the official java-based scorer of the WSD Evaluation Framework. """

    if test_set in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        gold_fn = 'ALL/ALL.gold.%s.key.txt' % test_set
    else:
        gold_fn = '%s/%s.gold.key.txt' % (test_set, test_set)

    cmd = 'cd %s && java Scorer %s %s' % (eval_fw_path + 'Evaluation_Datasets/',
                                          gold_fn,
                                          '../../../../' + results_path)
    print(cmd)
    cmd_output = subprocess.check_output(cmd, shell=True)
    cmd_output = cmd_output.decode('utf8')
    return cmd_output


def chunks(l, n):
    """Yield successive n-sized chunks from given list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def str_configuration(args):

    str_conf = args.lmms_path.split('/')[-1].split('.')[0]
    str_conf += '.%s' % args.test_set
    str_conf += '.%s' % args.nlm_id

    str_conf += '.%s' % args.layer_op
    if len(args.layers) > 1:
        str_conf += '.%d,%d' % (args.layers[0], args.layers[-1])
    else:
        str_conf += '.%d' % args.layers[0]

    return str_conf


def eval_wsd(args, encoder, eval_instances):

    """
    Initialize various counters for calculating supplementary metrics.
    """
    n_instances, n_sents, n_correct, n_unk_lemmas = 0, 0, 0, 0

    """
    Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
    File with predictions is processed by the official scorer after iterating over all instances.
    """

    str_conf = str_configuration(args)
    
    results_path = 'results/%s.key' % str_conf
    results_f = open(results_path, 'w')

    # matches_path = 'matches/%s.tsv' % str_conf
    # matches_f = open(matches_path, 'w')

    for batch in chunks(eval_instances, args.batch_size):
        
        batch_tokens = [e['tokens'] for e in batch]
        batch_embs = encoder.token_embeddings(batch_tokens)

        for sent_info, sent_embs in zip(batch, batch_embs):

            n_sents += 1
            idx_map_abs = sent_info['idx_map_abs']

            for mw_idx, tok_idxs in idx_map_abs:
                curr_sense = sent_info['senses'][mw_idx]

                if curr_sense is None:
                    continue

                n_instances += 1

                curr_lemma = sent_info['lemmas'][mw_idx]
                if curr_lemma not in senses_vsm.known_lemmas:
                    n_unk_lemmas += 1
                    continue  # skips hurt performance in official scorer

                curr_postag = sent_info['pos'][mw_idx]
                curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
                curr_vector = np.array([sent_embs[i][1] for i in tok_idxs]).mean(axis=0)
                curr_vector = curr_vector / np.linalg.norm(curr_vector)

                """
                Compose test-time embedding for matching with sense embeddings in SensesVSM.
                Test-time embedding may correspond to stack of contextual embeddings.
                Stacking composition performed according to dimensionality of sense embeddings.
                """

                # duplicating contextual feature for cos similarity against features from
                # sense annotations and glosses that belong to the same NLM
                if senses_vsm.ndims == 1024+1024:  # TODO: get dims from loaded model
                    curr_vector = np.hstack((curr_vector, curr_vector))

                elif senses_vsm.ndims == 4096+4096:
                    curr_vector = np.hstack((curr_vector, curr_vector))

                curr_vector = curr_vector / np.linalg.norm(curr_vector)

                """
                Matches test-time embedding against sense embeddings in SensesVSM.
                Matching is actually cosine similarity (most similar), or 1-NN.
                """
                matches = senses_vsm.match_senses(curr_vector, curr_lemma, curr_postag, topn=None)
                preds = [sk for sk, _ in matches]

                if len(preds) > 0:
                    results_f.write('%s %s\n' % (curr_sense, preds[0]))
                
                # matches_str = '\t'.join(['%s|%.5f' % (sk, sim) for sk, sim in matches[:100]])
                # matches_f.write('%s\t%s\t%s\n' % (curr_sense, ','.join(id2gold[curr_sense]), matches_str))

                gold_sensekeys = set(id2gold[curr_sense])
                if preds[0] in gold_sensekeys:
                    n_correct += 1

                if args.debug and (n_instances % 100 == 0):
                    acc = n_correct / n_instances
                    logging.info('ACC: %.3f (inst_idx=%d sent_idx=%d/%d)' % (acc, n_instances, n_sents, len(eval_instances)))

    results_f.close()
    # matches_f.close()

    if args.debug:
        logging.info('Final Results:')
        acc = n_correct / n_instances
        logging.info('ACC: %.3f (inst_idx=%d sent_idx=%d/%d)' % (acc, n_instances, n_sents, len(eval_instances)))
        logging.info('# unknown lemmas: %d' % n_unk_lemmas)

    logging.info('Running official scorer ...')
    scores_str = run_scorer(args.eval_fw_path, args.test_set, results_path)
    print(scores_str)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-test_set', default='ALL', help='Name of test set', required=False,
                        choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL', 'validation', 'NOUN', 'VERB', 'ADJ', 'ADV'])
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=True)
    parser.add_argument('-lmms_path', help='Path to LMMS sense embeddings', required=True)
    parser.add_argument('-sense_level', help='WN sense level', required=False, default='sensekey', choices=['sensekey', 'synset'])
    parser.add_argument('-weights_path', type=str, default='', help='Path to layer weights', required=False)
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False, choices=['mean', 'max', 'sum', 'concat', 'ws'])
    parser.add_argument('-subword_op', type=str, default='mean', help='Subword Reconstruction Strategy', required=False, choices=['mean', 'first', 'sum'])
    parser.add_argument('-eval_fw_path', help='Path to WSD Evaluation Framework', required=False, default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length', required=False)
    parser.set_defaults(debug=True)
    args = parser.parse_args()

    if args.nlm_id not in args.lmms_path.split('/')[-1].split('.'):  # catch mismatched nlms/sense_vecs
        logging.fatal("Provided sense embeddings don't seem to match nlm_id (%s)." % args.nlm_id)
        raise SystemExit('Fatal Error.')

    if args.layer_op == 'ws' and args.weights_path == '':
        raise(BaseException('Weights path must be given with layer_op \'ws\''))

    if '-base' in args.nlm_id:
        nmax_layers = 12
    elif args.nlm_id.startswith('albert-xxlarge'):  # exception for albert-xxlarge (to refactor later...)-
        nmax_layers = 12
    else:
        nmax_layers = 24

    if args.layer_op == 'ws':
        args.layers = 'all'  # override

    if args.layers == 'all':
        args.layers = [-n for n in range(1, nmax_layers + 1)]
    else:
        args.layers = [int(n) for n in args.layers.split(' ')]

    """
    Load sense embeddings for evaluation.
    """
    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.lmms_path, normalize=True)
    logging.info('Loaded SensesVSM')

    """
    Load evaluation instances and gold labels.
    Gold labels (sensekeys) only used for reporting accuracy during evaluation.
    """

    if args.test_set in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        wsd_fw_set_path = args.eval_fw_path + 'Evaluation_Datasets/ALL/ALL.data.xml'
        wsd_fw_gold_path = args.eval_fw_path + 'Evaluation_Datasets/ALL/ALL.gold.%s.key.txt' % args.test_set
    else:
        wsd_fw_set_path = args.eval_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
        wsd_fw_gold_path = args.eval_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)
    
    id2gold = load_ground_truth(wsd_fw_gold_path, sense_level=args.sense_level)
    eval_instances = load_wsd_fw_set(wsd_fw_set_path)

    encoder_cfg = {
        'model_name_or_path': args.nlm_id,
        'min_seq_len': 0,
        'max_seq_len': args.max_seq_len,
        'layers': args.layers,
        'layer_op': args.layer_op,
        'weights_path': args.weights_path,
        'subword_op': args.subword_op
    }

    if encoder_cfg['model_name_or_path'].split('-')[0] in ['roberta', 'xlmr']:
        encoder = FairSeqEncoder(encoder_cfg)
    else:
        encoder = TransformersEncoder(encoder_cfg)

    """
    Pre-processing mapping between sensekeys and synsets.
    """
    map_sk2syn = {}
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            map_sk2syn[lemma.key()] = synset.name()

    eval_wsd(args, encoder, eval_instances)
