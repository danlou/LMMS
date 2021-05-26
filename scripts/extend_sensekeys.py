import logging
import argparse
from functools import lru_cache
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn

import sys, os  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from vectorspace import SensesVSM

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def get_sensekey2synset_map():
    sensekey2synset_map = {}
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            sensekey2synset_map[lemma.key()] = synset
    return sensekey2synset_map

sensekey2synset_map = get_sensekey2synset_map()

all_sensekeys = list(sensekey2synset_map.keys())


def wn_sensekey2synset(sensekey):
    return sensekey2synset_map[sensekey]


@lru_cache()
def wn_synset2sensekeys(synset):
    sensekeys = []
    for lemma in synset.lemmas():
        sensekeys.append(lemma.key())
    return sensekeys


def get_synset_vec(senses_vsm, synset, additional_vecs={}):
    sk_vecs = []
    for synset_sensekey in wn_synset2sensekeys(synset):
        if synset_sensekey in senses_vsm.labels_set:
            sk_vecs.append(senses_vsm.get_vec(synset_sensekey))
        elif synset_sensekey in additional_vecs:
            sk_vecs.append(additional_vecs[synset_sensekey])

    if len(sk_vecs) > 0:
        return np.mean(sk_vecs, axis=0)
    else:
        return None


def wn_all_lexnames():
    all_lexs = set()
    for s in wn.all_synsets():
        all_lexs.add(s.lexname())
    return all_lexs


def wn_all_lexnames_groups():
    groups = defaultdict(list)
    for synset in wn.all_synsets():
        groups[synset.lexname()].append(synset)
    return dict(groups)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Propagates supervised sense embeddings through WordNet.')
    parser.add_argument('-sup_sv_path', help='Path to supervised sense vectors', required=True)
    parser.add_argument('-ext_mode', default='lexname', help='Max abstraction level', required=False,
                        choices=['synset', 'hypernym', 'lexname'])
    parser.add_argument('-out_path', help='Path to resulting extended vector set', required=True)
    args = parser.parse_args()

    n_total_senses = len(all_sensekeys)

    logging.info('Loading SensesVSM ...')
    senses_vsm = SensesVSM(args.sup_sv_path, normalize=False)

    logging.info('Processing ...')
    additional_vecs = defaultdict(list)

    sks_represented = set(senses_vsm.labels_set)

    if args.ext_mode in ['synset', 'hypernym', 'lexname']:
        # synset expansion
        n_added_by_synset = 0
        for sensekey_idx, sensekey in enumerate(all_sensekeys):
            if sensekey in sks_represented:
                continue

            synset = wn_sensekey2synset(sensekey)
            synset_vec = get_synset_vec(senses_vsm, synset, additional_vecs)
            if synset_vec is not None:
                additional_vecs[sensekey] = synset_vec
                sks_represented.add(sensekey)
                n_added_by_synset += 1
                # continue
        
        logging.info('Added %d sks by synset propagation.' % n_added_by_synset)


    if args.ext_mode in ['hypernym', 'lexname']:
        # hypernym expansion
        n_added_by_hypernym = 0
        for sensekey_idx, sensekey in enumerate(all_sensekeys):
            if sensekey in sks_represented:
                continue

            synset = wn_sensekey2synset(sensekey)
            synset_vec = get_synset_vec(senses_vsm, synset, additional_vecs)

            hypernym_vecs = []
            for hypernym_synset in set(synset.hypernyms() + synset.instance_hypernyms()):
                hypernym_synset_vec = get_synset_vec(senses_vsm, hypernym_synset, additional_vecs)
                if hypernym_synset_vec is not None:
                    hypernym_vecs.append(hypernym_synset_vec)

            if len(hypernym_vecs) > 0:
                additional_vecs[sensekey] = np.mean(hypernym_vecs, axis=0)
                sks_represented.add(sensekey)
                n_added_by_hypernym += 1
                # continue
        
        logging.info('Added %d sks by hypernym propagation.' % n_added_by_hypernym)

    if args.ext_mode in ['lexname']:
        logging.info('Preparing lexname vecs ...')
        lexname_vecs = defaultdict(list)
        lexname_groups = wn_all_lexnames_groups()
        for lexname, synsets in lexname_groups.items():
            lexname_groups_vecs = [get_synset_vec(senses_vsm, s, additional_vecs) for s in synsets]
            lexname_groups_vecs = [v for v in lexname_groups_vecs if v is not None]
            if len(lexname_groups_vecs) > 0:
                lexname_vecs[lexname] = np.mean(lexname_groups_vecs, axis=0)
            else:
                logging.warning('No vecs for lexname %d' % lexname)

        # lexname expansion
        n_added_by_lexname = 0
        for sensekey_idx, sensekey in enumerate(all_sensekeys):
            if sensekey in sks_represented:
                continue

            synset = wn_sensekey2synset(sensekey)

            if synset.lexname() in lexname_vecs:
                additional_vecs[sensekey] = lexname_vecs[synset.lexname()]
                sks_represented.add(sensekey)
                n_added_by_lexname += 1
            else:
                logging.warning('Missing lexname %s' % synset)

        logging.info('Added %d sks by lexname propagation.' % n_added_by_lexname)

        # logging.info('Lexname Counts:')
        # for lexname_idx, lexname in enumerate(wn_all_lexnames()):
        #     logging.info(lexname_idx, lexname, len(lexname_groups[lexname]))

    # write vecs
    logging.info('Writing %s ...' % args.out_path)
    n_vecs = len(senses_vsm.vectors) + len(additional_vecs)
    assert n_vecs == len(sks_represented)

    n_total_senses = len(all_sensekeys)
    logging.info('n_vecs: %d - %d' % (n_vecs, n_total_senses))
    logging.info('Coverage: %f' % (n_vecs/n_total_senses))

    with open(args.out_path, 'w') as extended_f:

        # logging.info('Writing provided vecs ...')
        with open(args.sup_sv_path) as supervised_f:
            for line in supervised_f:
                extended_f.write(line)

        # logging.info('Writing inferred vecs ...')
        for sensekey, vec in additional_vecs.items():
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            extended_f.write('%s %s\n' % (sensekey, vec_str))

    logging.info('Done')