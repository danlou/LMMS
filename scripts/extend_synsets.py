import logging
import argparse
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn

import sys, os  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from vectorspace import VSM

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


all_synsets = [synset.name() for synset in wn.all_synsets()]


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
                        choices=['hypernym', 'lexname'])
    parser.add_argument('-out_path', help='Path to resulting extended vector set', required=True)
    args = parser.parse_args()

    n_total_senses = len(all_synsets)

    logging.info('Loading VSM ...')
    vsm = VSM()
    vsm.load_txt(args.sup_sv_path)

    logging.info('Processing ...')
    additional_vecs = defaultdict(list)

    syns_represented = set(vsm.labels_set)

    if args.ext_mode in ['hypernym', 'lexname']:
        # hypernym expansion
        n_added_by_hypernym = 0
        for synset_idx, synset in enumerate(all_synsets):
            if synset in syns_represented:
                continue

            hypernym_vecs = []
            for hypernym in set(wn.synset(synset).hypernyms() + wn.synset(synset).instance_hypernyms()):
                hypernym = hypernym.name()
                if hypernym in vsm.labels_set:
                    hypernym_vecs.append(vsm.get_vec(hypernym))

            if len(hypernym_vecs) > 0:
                additional_vecs[synset] = np.mean(hypernym_vecs, axis=0)
                syns_represented.add(synset)
                n_added_by_hypernym += 1
        
        logging.info('Added %d syns by hypernym propagation.' % n_added_by_hypernym)

    if args.ext_mode in ['lexname']:
        logging.info('Preparing lexname vecs ...')
        all_lexname_vecs = {}
        lexname_groups = wn_all_lexnames_groups()
        for lexname, lexname_synsets in lexname_groups.items():
            lexname_vecs =  [vsm.get_vec(s.name()) for s in lexname_synsets if s.name() in vsm.labels_set]
            lexname_vecs += [additional_vecs[s.name()] for s in lexname_synsets if s.name() in additional_vecs]

            if len(lexname_vecs) > 0:
                all_lexname_vecs[lexname] = np.mean(lexname_vecs, axis=0)
            else:
                logging.warning('No vecs for lexname %d' % lexname)

        # lexname expansion
        n_added_by_lexname = 0
        for synset_idx, synset in enumerate(all_synsets):
            if synset in syns_represented:
                continue
            
            lexname = wn.synset(synset).lexname()
            if lexname in all_lexname_vecs:
                additional_vecs[synset] = all_lexname_vecs[lexname]
                syns_represented.add(synset)
                n_added_by_lexname += 1
            else:
                logging.warning('Missing lexname %s' % synset)

        logging.info('Added %d sks by lexname propagation.' % n_added_by_lexname)

        # logging.info('Lexname Counts:')
        # for lexname_idx, lexname in enumerate(wn_all_lexnames()):
        #     logging.info(lexname_idx, lexname, len(lexname_groups[lexname]))

    # write vecs
    logging.info('Writing vecs ...')
    n_vecs = len(vsm.vectors) + len(additional_vecs)
    assert n_vecs == len(syns_represented)

    n_total_senses = len(all_synsets)
    logging.info('n_vecs: %d - %d' % (n_vecs, n_total_senses))
    logging.info('Coverage: %f' % (n_vecs/n_total_senses))

    with open(args.out_path, 'w') as extended_f:

        with open(args.sup_sv_path) as supervised_f:
            for line in supervised_f:
                extended_f.write(line)

        for synset, vec in additional_vecs.items():
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            extended_f.write('%s %s\n' % (synset, vec_str))
