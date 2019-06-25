import os
import argparse
import logging
from time import time
from functools import lru_cache
from datetime import datetime

import numpy as np
from nltk.corpus import wordnet as wn

import spacy
nlp = spacy.load('en_core_web_sm')

from bert_as_service import bert_embed
from vectorspace import SensesVSM


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


@lru_cache()
def wn_sensekey2synset(sensekey):
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


def get_sent_info(merge_ents=False):
    sent_info = {'tokens': [], 'lemmas': [], 'pos': [], 'sentence': ''}

    sent_info['sentence'] = input('Input Sentence (\'q\' to exit):\n')

    doc = nlp(sent_info['sentence'])

    if merge_ents:
        for ent in doc.ents:
            ent.merge()

    for tok in doc:
        sent_info['tokens'].append(tok.text.replace(' ', '_'))
        # sent_info['tokens'].append(tok.text)
        sent_info['lemmas'].append(tok.lemma_)
        sent_info['pos'].append(tok.pos_)

    sent_info['tokenized_sentence'] = ' '.join(sent_info['tokens'])

    return sent_info


def map_senses(svsm, tokens, postags=[], lemmas=[], use_postag=False, use_lemma=False):
    """Given loaded LMMS and a list of tokens, returns a list of scored sensekeys."""

    matches = []

    if len(tokens) != len(postags):  # mismatched
        use_postag = False

    if len(tokens) != len(lemmas):  # mismatched
        use_lemma = False

    sent_bert = bert_embed([' '.join(tokens)], merge_strategy='mean')[0]

    for idx in range(len(tokens)):
        idx_vec = sent_bert[idx][1]
        idx_vec = idx_vec / np.linalg.norm(idx_vec)

        if svsm.ndims == 1024:
            # idx_vec = idx_vec
            pass

        elif svsm.ndims == 1024+1024:
            idx_vec = np.hstack((idx_vec, idx_vec))
            idx_vec = idx_vec / np.linalg.norm(idx_vec)

        idx_matches = []
        if use_lemma and use_postag:
            idx_matches = svsm.match_senses(idx_vec, lemmas[idx], postags[idx], topn=None)

        elif use_lemma:
            idx_matches = svsm.match_senses(idx_vec, lemmas[idx], None, topn=None)

        elif use_postag:
            idx_matches = svsm.match_senses(idx_vec, None, postags[idx], topn=None)

        else:
            idx_matches = svsm.match_senses(idx_vec, None, None, topn=None)

        matches.append(idx_matches)

    return matches



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concept Mapping Demonstration.')
    parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
    args = parser.parse_args()

    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.sv_path, normalize=True)

    while True:
        sent_info = get_sent_info()

        if sent_info['sentence'] == 'q':
            break
        elif len(sent_info['sentence']) == 0:
            continue

        matches = map_senses(senses_vsm,
                             sent_info['tokens'],
                             sent_info['pos'],
                             sent_info['lemmas'],
                             use_lemma=False,
                             use_postag=False)

        for idx, idx_matches in enumerate(matches):
            print()
            print('TOK: %s | POS: %s | LEM: %s' % (sent_info['tokens'][idx],
                                                   sent_info['pos'][idx],
                                                   sent_info['lemmas'][idx]))

            print('Top 10 Matches (out of %d):' % len(idx_matches))
            for sk_idx, (sk, score) in enumerate(idx_matches[:10]):
                synset = wn_sensekey2synset(sk)
                print('#%d - %.3f %s %s' % (sk_idx + 1, score, sk, synset))
                print('DEF: %s' % synset.definition())
                print()
