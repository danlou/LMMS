"""
Example usage (from root folder):
python evaluation/eval_wic.py -eval_set dev -nlm_id albert-xxlarge-v2 -lmms_path data/vectors/albert-xxlarge-v2/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt -weights_path data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt -layer_op ws
"""

import os
import argparse
import logging
from functools import lru_cache

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wn_lemmatizer = WordNetLemmatizer()

import sys  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder

from vectorspace import SensesVSM


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


@lru_cache()
def wn_sensekey2synset(sensekey):
    """ Convert sensekey to synset. """
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
    parser = argparse.ArgumentParser(description='Evaluation of WiC solution using LMMS for sense comparison.')
    parser.add_argument('-eval_set', default='dev', help='Evaluation set', required=False, choices=['train', 'dev', 'test'])
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-large-cased')
    parser.add_argument('-lmms_path', help='Path to LMMS vectors', required=True)
    parser.add_argument('-weights_path', type=str, default='', help='Path to layer weights', required=False)
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False, choices=['mean', 'max', 'sum', 'concat', 'ws'])
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length', required=False)
    parser.add_argument('-subword_op', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False, choices=['mean', 'first', 'sum'])
    args = parser.parse_args()

    if args.layer_op == 'ws' and args.weights_path == '':
        raise(BaseException('Weights path must be given with layer_op \'ws\''))

    if args.layer_op == 'ws':
        args.layers = 'all'  # override

    if args.layers == 'all':
        if '-base' in args.nlm_id or args.nlm_id == 'albert-xxlarge-v2':
            nmax_layers = 12 + 1
        else:
            nmax_layers = 24 + 1
        args.layers = [-n for n in range(1, nmax_layers + 1)]
    else:
        args.layers = [int(n) for n in args.layers.split(' ')]

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

    results_path = 'results/wic.compare.%s.txt' % args.eval_set

    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.lmms_path, normalize=True)


    logging.info('Processing sentences ...')
    n_instances, n_correct = 0, 0
    with open(results_path, 'w') as results_f:  # store results in WiC's format
        for wic_idx, wic_entry in enumerate(load_wic(args.eval_set, wic_path='external/wic')):
            word, postag, idx1, idx2, ex1, ex2, gold = wic_entry

            embs_ex1, embs_ex2 = encoder.token_embeddings([ex1.split(), ex2.split()])

            # example1
            ex1_curr_word, ex1_curr_vector = embs_ex1[idx1]
            ex1_curr_lemma = wn_lemmatize(word, postag)
            ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)

            if senses_vsm.ndims in [1024+1024, 4096+4096]:  # concatenated
                ex1_curr_vector = np.hstack((ex1_curr_vector, ex1_curr_vector))
                ex1_curr_vector = ex1_curr_vector / np.linalg.norm(ex1_curr_vector)

            ex1_matches = senses_vsm.match_senses(ex1_curr_vector, lemma=ex1_curr_lemma, postag=postag, topn=None)
            ex1_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex1_matches]
            ex1_wsd_vector = senses_vsm.get_vec(ex1_matches[0][0])

            # example2
            ex2_curr_word, ex2_curr_vector = embs_ex2[idx2]
            ex2_curr_lemma = wn_lemmatize(word, postag)
            ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)

            if senses_vsm.ndims in [1024+1024, 4096+4096]:  # concatenated
                ex2_curr_vector = np.hstack((ex2_curr_vector, ex2_curr_vector))
                ex2_curr_vector = ex2_curr_vector / np.linalg.norm(ex2_curr_vector)

            ex2_matches = senses_vsm.match_senses(ex2_curr_vector, lemma=ex2_curr_lemma, postag=postag, topn=None)
            ex2_synsets = [(wn_sensekey2synset(sk), score) for sk, score in ex2_matches]
            ex2_wsd_vector = senses_vsm.get_vec(ex2_matches[0][0])

            ex1_best = ex1_synsets[0][0]
            ex2_best = ex2_synsets[0][0]

            n_instances += 1

            identical = False
            if len(ex1_synsets) == 1:
                identical = True

            elif ex1_best == ex2_best:
                identical = True

            elif ex1_best != ex2_best:
                identical = False

            if identical:
                results_f.write('T\n')
            else:
                results_f.write('F\n')

            if identical == gold:
                n_correct += 1

            acc = n_correct/n_instances
            logging.info('ACC: %f (%d/%d)' % (acc, n_correct, n_instances))

logging.info('Saved predictions to %s' % results_path)
