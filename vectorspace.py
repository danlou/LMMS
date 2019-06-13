from time import time
from functools import lru_cache
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn


def get_sk_type(sensekey):
    return int(sensekey.split('%')[1].split(':')[0])


def get_sk_pos(sk, tagtype='long'):
    # merges ADJ with ADJ_SAT

    if tagtype == 'long':
        type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
        return type2pos[get_sk_type(sk)]

    elif tagtype == 'short':
        type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
        return type2pos[get_sk_type(sk)]


def get_sk_lemma(sensekey):
    return sensekey.split('%')[0]


class SensesVSM(object):

    def __init__(self, vecs_path, normalize=True):
        self.vecs_path = vecs_path
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        if self.vecs_path.endswith('.txt'):
            self.load_txt(self.vecs_path)

        elif self.vecs_path.endswith('.npz'):
            self.load_npz(self.vecs_path)

        self.load_aux_senses()

        if normalize:
            self.normalize()

    def load_txt(self, txt_vecs_path):
        self.vectors = []
        with open(txt_vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
        self.vectors = np.vstack(self.vectors)

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def load_npz(self, npz_vecs_path):
        loader = np.load(npz_vecs_path)
        self.labels = loader['labels'].tolist()
        self.vectors = loader['vectors']

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def load_aux_senses(self):

        self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
        self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}

        self.lemma_sks = defaultdict(list)
        for sk, lemma in self.sk_lemmas.items():
            self.lemma_sks[lemma].append(sk)
        self.known_lemmas = set(self.lemma_sks.keys())

        self.sks_by_pos = defaultdict(list)
        for s in self.labels:
            self.sks_by_pos[self.sk_postags[s]].append(s)
        self.known_postags = set(self.sks_by_pos.keys())

    def save_npz(self):
        npz_path = self.vecs_path.replace('.txt', '.npz')
        np.savez_compressed(npz_path,
                            labels=self.labels,
                            vectors=self.vectors)

    def normalize(self, norm='l2'):
        norms = np.linalg.norm(self.vectors, axis=1)
        self.vectors = (self.vectors.T / norms).T

    def get_vec(self, label, as_numpy=True):
        if as_numpy:
            return np.asnumpy(self.vectors[self.indices[label]])
        else:
            return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def match_senses(self, vec, lemma=None, postag=None, topn=100):

        relevant_sks = []
        for sk in self.labels:
            if (lemma is None) or (self.sk_lemmas[sk] == lemma):
                if (postag is None) or (self.sk_postags[sk] == postag):
                    relevant_sks.append(sk)
        relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]

        sims = np.dot(self.vectors[relevant_sks_idxs], np.array(vec))
        matches = list(zip(relevant_sks, sims))

        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        return matches[:topn]

    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.vectors, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()


class VSM(object):

    def __init__(self, vecs_path, normalize=True):
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        self.load_txt(vecs_path)

        if normalize:
            self.normalize()

    def load_txt(self, vecs_path):
        # print('Loading VSM ...', end=' ')
        self.vectors = []
        with open(vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))

                # if line_idx % 100000 == 0:
                #     print(line_idx)

        self.vectors = np.vstack(self.vectors)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]
        # print('Done')

    def normalize(self, norm='l2'):
        self.vectors = (self.vectors.T / np.linalg.norm(self.vectors, axis=1)).T

    def get_vec(self, label, as_numpy=True):
        if as_numpy:
            return np.asnumpy(self.vectors[self.indices[label]])
        else:
            return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.vectors, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()
