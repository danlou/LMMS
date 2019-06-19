import os
import argparse
import logging
from functools import lru_cache
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wn_lemmatizer = WordNetLemmatizer()

from sklearn.externals import joblib

import sys  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from bert_as_service import bert_embed
from vectorspace import SensesVSM


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


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
def wn_lemmatize(w, postag=None):
    w = w.lower()
    if postag is not None:
        return wn_lemmatizer.lemmatize(w, pos=postag[0].lower())
    else:
        return wn_lemmatizer.lemmatize(w)


def load_wic(setname='dev', wic_path='external/wic'):
    data_entries = []
    pos_map = {'N': 'NOUN', 'V': 'VERB'}
    data_path = '%s/%s/%s.data.txt' % (wic_path, setname, setname)
    for line in open(data_path):
        word, pos, idxs, ex1, ex2 = line.strip().split('\t')
        idx1, idx2 = list(map(int, idxs.split('-')))
        data_entries.append([word, pos_map[pos], idx1, idx2, ex1, ex2])

    if setname == 'test':  # no gold
        return [e + [None] for e in data_entries]

    gold_entries = []
    gold_path = '%s/%s/%s.gold.txt' % (wic_path, setname, setname)
    for line in open(gold_path):
        gold = line.strip()
        if gold == 'T':
            gold_entries.append(True)
        elif gold == 'F':
            gold_entries.append(False)

    assert len(data_entries) == len(gold_entries)
    return [e + [gold_entries[i]] for i, e in enumerate(data_entries)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of WiC solution using LMMS and LogReg for binary classification.')
    parser.add_argument('-lmms_path', help='Path to LMMS vectors', required=True)
    parser.add_argument('-eval_set', default='dev', help='Evaluation set', required=False, choices=['train', 'dev', 'test'])
    parser.add_argument('-clf_path', help='Path to .pkl LR classifier', required=False,
                        default='data/models/wic.lr_4feats_1556300807.pkl')
    args = parser.parse_args()

    logging.info('Loading LR Classifer ...')
    clf = joblib.load(args.clf_path)

    results_path = 'data/results/wic.classify.%s.txt' % args.eval_set

    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.lmms_path, normalize=True)

    logging.info('Processing sentences ...')
    n_instances, n_correct = 0, 0
    with open(results_path, 'w') as results_f:  # store results in WiC's format
        for wic_idx, wic_entry in enumerate(load_wic(args.eval_set, wic_path='external/wic')):
            word, postag, idx1, idx2, ex1, ex2, gold = wic_entry

            bert_ex1, bert_ex2 = bert_embed([ex1, ex2], merge_strategy='mean')

            # example1
            ex1_curr_word, ex1_curr_vector = bert_ex1[idx1]
            ex1_curr_lemma = wn_lemmatize(word, postag)
            ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)

            if senses_vsm.ndims == 1024:
                ex1_curr_vector = ex1_curr_vector

            elif senses_vsm.ndims == 1024+1024:
                ex1_curr_vector = np.hstack((ex1_curr_vector, ex1_curr_vector))

            ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)
            ex1_matches = senses_vsm.match_senses(ex1_curr_vector, lemma=ex1_curr_lemma, postag=postag, topn=None)
            ex1_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex1_matches]
            ex1_wsd_vector = senses_vsm.get_vec(ex1_matches[0][0])

            # example2
            ex2_curr_word, ex2_curr_vector = bert_ex2[idx2]
            ex2_curr_lemma = wn_lemmatize(word, postag)
            ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)

            if senses_vsm.ndims == 1024:
                ex2_curr_vector = ex2_curr_vector

            elif senses_vsm.ndims == 1024+1024:
                ex2_curr_vector = np.hstack((ex2_curr_vector, ex2_curr_vector))

            ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)
            ex2_matches = senses_vsm.match_senses(ex2_curr_vector, lemma=ex2_curr_lemma, postag=postag, topn=None)
            ex2_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex2_matches]
            ex2_wsd_vector = senses_vsm.get_vec(ex2_matches[0][0])

            n_instances += 1
            s1_sim = np.dot(ex1_curr_vector, ex2_curr_vector)
            s2_sim = np.dot(ex1_wsd_vector, ex2_wsd_vector)
            s3_sim = np.dot(ex1_curr_vector, ex1_wsd_vector)
            s4_sim = np.dot(ex2_curr_vector, ex2_wsd_vector)

            # pred = clf.predict([[s1_sim]])[0]
            # pred = clf.predict([[s2_sim]])[0]
            # pred = clf.predict([[s1_sim, s2_sim]])[0]
            # pred = clf.predict([[s2_sim, s3_sim, s4_sim]])[0]
            pred = clf.predict([[s1_sim, s2_sim, s3_sim, s4_sim]])[0]

            if pred:
                results_f.write('T\n')
            else:
                results_f.write('F\n')

            if pred == gold:
                n_correct += 1
            # else:
            #     print('WRONG')

            # print(wic_idx, pred, gold)
            # print(word, postag, idx1, idx2, ex1, ex2, gold)
            # print(n_correct/n_instances, n_correct, n_instances)
            # print()

            acc = n_correct/n_instances
            logging.info('ACC: %f (%d/%d)' % (acc, n_correct, n_instances))

logging.info('Saved predictions to %s' % results_path)