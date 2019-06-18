import os
import logging
import argparse
from time import time
from functools import lru_cache
from collections import defaultdict
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@lru_cache()
def wn_sensekey2synset(sensekey):
    """Convert sensekey to synset."""
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


@lru_cache()
def wn_first_sense(lemma, postag=None):
    pos_map = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}
    first_synset = wn.synsets(lemma, pos=pos_map[postag])[0]
    found = False
    for lem in first_synset.lemmas():
        key = lem.key()
        if key.startswith('{}%'.format(lemma)):
            found = True
            break
    assert found
    return key


def load_wsd_fw_set(eval_path):
    """Parse XML of split set and return list of instances (dict)."""
    eval_entries = []
    tree = ET.parse(eval_path)
    for text in tree.getroot():
        for sent_idx, sentence in enumerate(text):
            sent_info = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
            for e in sentence:
                sent_info['tokens_mw'].append(e.text)
                sent_info['lemmas'].append(e.get('lemma'))
                sent_info['senses'].append(e.get('id'))
                sent_info['pos'].append(e.get('pos'))

            sent_info['tokens'] = sum([t.split() for t in sent_info['tokens_mw']], [])

            # handling multi-word expressions, mapping allows matching tokens with mw features
            idx_map_abs = []
            idx_map_rel = [(i, list(range(len(t.split()))))
                            for i, t in enumerate(sent_info['tokens_mw'])]
            token_counter = 0
            for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                idx_tokens = [i+token_counter for i in idx_tokens]
                token_counter += len(idx_tokens)
                idx_map_abs.append([idx_group, idx_tokens])

            sent_info['tokenized_sentence'] = ' '.join(sent_info['tokens'])
            sent_info['idx_map_abs'] = idx_map_abs
            sent_info['idx'] = sent_idx

            eval_entries.append(sent_info)

    return eval_entries


def get_id2senses(wsd_eval_keys):
    """Maps ids of split set to sensekeys, just for in-code evaluation."""
    id2senses = {}
    with open(wsd_eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2senses[id_] = keys
    return id2senses


def run_scorer(eval_framework_path, eval_set, results_path):
    """Runs the official java-based scorer of the WSD Evaluation Framework."""
    cmd = 'cd %s && java Scorer %s %s' % (eval_framework_path + 'Evaluation_Datasets/',
                                          '%s/%s.gold.key.txt' % (eval_set, eval_set),
                                          '../../../../' + results_path)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Most Frequent Sense (i.e. 1st) evaluation of WSD Evaluation Framework.')
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework.', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-test_set', default='ALL', help='Name of test set', required=False,
                        choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
    args = parser.parse_args()

    results_path = 'data/results/%d.%s.mfs.key' % (int(time()), args.test_set)

    wsd_eval_data = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
    wsd_eval_keys = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)

    debug = True  # flag to print wrong results in progress

    # counters for additional evaluation
    id2senses = get_id2senses(wsd_eval_keys)  # for ACC in progress
    n_instances, n_correct = 0, 0
    failed_by_pos = defaultdict(list)

    eval_entries = load_wsd_fw_set(wsd_eval_data)

    with open(results_path, 'w') as results_f:  # store results in WSD_Evaluation_Framework's format
        for batch_idx, batch in enumerate(chunks(eval_entries, len(eval_entries))):
            batch_t0 = time()
            batch_sents = [sent_info['tokenized_sentence'] for sent_info in batch]

            for sent_info in batch:
                idx_map_abs = sent_info['idx_map_abs']

                for mw_idx, tok_idxs in idx_map_abs:
                    curr_sense = sent_info['senses'][mw_idx]

                    if curr_sense is None:
                        continue

                    curr_lemma = sent_info['lemmas'][mw_idx]
                    curr_postag = sent_info['pos'][mw_idx]

                    matches = [(wn_first_sense(curr_lemma, curr_postag), 1)]

                    preds = [sensekey for sensekey, sim in matches]

                    if len(preds) > 0:
                        results_f.write('%s %s\n' % (curr_sense, preds[0]))

                    if debug:
                        n_instances += 1
                        gold_sensekeys = id2senses[curr_sense]
                        gold_synsets = [wn_sensekey2synset(k).name() for k in gold_sensekeys]

                        wsd_correct = False
                        if len(set(preds).intersection(set(gold_sensekeys))) > 0:
                            n_correct += 1
                            wsd_correct = True
                        else:
                            if len(preds) > 0:
                                failed_by_pos[curr_postag].append((preds[0], gold_sensekeys))
                            else:
                                failed_by_pos[curr_postag].append((None, gold_sensekeys))

                        acc = n_correct / n_instances
                        logging.debug('ACC: %.3f (%d/%d)' % (acc, n_correct, n_instances))

    for pos, fails in failed_by_pos.items():
        logging.info('%s fails: %d' % (pos, len(fails)))

    logging.info('Running official scorer ...')
    run_scorer(args.wsd_fw_path, args.test_set, results_path)
