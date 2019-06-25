import argparse
import logging
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.switch_backend('agg')
plt.rcParams.update({'font.size': 18})

from vectorspace import SensesVSM


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def wn_synset2sensekeys(synset):
    sensekeys = []
    for lemma in synset.lemmas():
        sensekeys.append(lemma.key())
    return sensekeys


def get_synset_vec(senses_vsm, synset):
    synset_vecs = []
    for synset_sensekey in wn_synset2sensekeys(synset):
        if synset_sensekey in senses_vsm.labels_set:
            synset_vecs.append(senses_vsm.get_vec(synset_sensekey))

    if len(synset_vecs) > 0:
        return np.mean(synset_vecs, axis=0)
    else:
        return None


def get_synset_name_tex(synset):
    name, pos, ind = synset.name().split('.')
    # return '$%s_{\small{%d}}^{\small{%s}}$' % (name, int(ind), pos)
    return '$%s_{\small{%s}}^{\small{%d}}$' % (name, pos, int(ind))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bias Discovery Experiment with LMMS.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lmms1024', help='Path to LMMS 1024 vectors', required=True)
    parser.add_argument('-lmms2048', help='Path to LMMS 2048 vectors', required=True)
    parser.add_argument('-gen_pdf', help='Generate Latex Chart', action='store_true', required=False)
    parser.set_defaults(gen_pdf=False)
    args = parser.parse_args()

    relevant_synsets = []
    relevant_synsets.append(wn.synset('programmer.n.01'))
    relevant_synsets.append(wn.synset('doctor.n.01'))
    relevant_synsets.append(wn.synset('doctor.n.04'))
    relevant_synsets.append(wn.synset('counselor.n.01'))
    relevant_synsets.append(wn.synset('counselor.n.02'))
    relevant_synsets.append(wn.synset('florist.n.01'))
    relevant_synsets.append(wn.synset('teacher.n.01'))
    relevant_synsets.append(wn.synset('nurse.n.01'))
    relevant_synsets.append(wn.synset('receptionist.n.01'))

    logging.info('Loading SensesVSM for LMMS 1024 ...')
    senses_vsm_1pt = SensesVSM(args.lmms1024)

    logging.info('Loading SensesVSM for LMMS 2048 ...')
    senses_vsm_2pt = SensesVSM(args.lmms2048)

    vsms = [senses_vsm_1pt, senses_vsm_2pt]
    
    if args.gen_pdf:
        plt.clf()
        fig, ax = plt.subplots()

        vsm_labels = ['LMMS$_{1024}$', 'LMMS$_{2048}$']
        vsm_colors = ['deepskyblue', 'royalblue']
        vsm_patterns = ['/', 'x']

    logging.info('Processing Sense Vectors ...')
    for sense_vsm_idx, senses_vsm in enumerate(vsms):

        vec_man = get_synset_vec(senses_vsm, wn.synset('man.n.01'))
        vec_woman = get_synset_vec(senses_vsm, wn.synset('woman.n.01'))

        scored_synsets = []
        for synset in relevant_synsets:
            vec_synset = get_synset_vec(senses_vsm, synset)
            score = np.dot(vec_man, vec_synset) - np.dot(vec_woman, vec_synset)
            logging.info('bias(%s) = %f' % (synset.name(), score))
            scored_synsets.append((synset, score))
        scored_synsets = sorted(scored_synsets, key=lambda x: x[1], reverse=True)

        if args.gen_pdf:
            ax.barh(np.arange(len(relevant_synsets)),
                    [score for synset, score in scored_synsets],
                    label=vsm_labels[sense_vsm_idx],
                    align='center',
                    alpha=0.5,
                    color=vsm_colors[sense_vsm_idx],
                    hatch=vsm_patterns[sense_vsm_idx])

    if args.gen_pdf:
        ax.set_yticks(np.arange(len(relevant_synsets)))
        ax.set_yticklabels([get_synset_name_tex(synset) for synset, score in scored_synsets])
        ax.invert_yaxis()
        # ax.set_xlabel('Gender Bias')
        plt.legend(loc='upper left')

        logging.info('Saving Bar Chart PDF at misc/bias.pdf ...')
        plt.savefig('misc/bias.pdf', bbox_inches='tight', format='pdf', dpi=300)
