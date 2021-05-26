import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from collections import defaultdict
import numpy as np
from nltk.corpus import wordnet as wn

from vectorspace import VSM

def syn2sks(synset):
    return list(set([lemma.key() for lemma in synset.lemmas()]))


sks_vecs_path  = sys.argv[1]
syns_vecs_path = sys.argv[2]

print('Loading sensekey vecs ...')
sks_vsm = VSM()
sks_vsm.load_txt(sks_vecs_path)


print('Aggregating synset vecs ...')
syn_vecs = defaultdict(list)
for syn in wn.all_synsets():
    for sk in syn2sks(syn):
        if sk in sks_vsm.labels_set:
            syn_vecs[syn.name()].append(sks_vsm.get_vec(sk))


print('Writing synset vecs ...')
with open(syns_vecs_path, 'w') as syns_vecs_f:
    for syn, syn_vecs in syn_vecs.items():
        syn_vec = np.array(syn_vecs).mean(axis=0)
        syn_vec_str = ' '.join([str(round(v, 6)) for v in syn_vec.tolist()])
        syns_vecs_f.write('%s %s\n' % (syn, syn_vec_str))

