"""
Wrapper for NLTK's WordNet with some auxilliary methods.
Work-in-progress (mostly untested, and not used in LMMS evaluations).
"""

from functools import lru_cache
from collections import Counter
from nltk.corpus import wordnet as wn


class NoSynset(Exception):
    # TODO
   pass


class WN_Utils():

    def __init__(self):
        self.map_sk2syn = {}
        self.load_sk2syn()

    def load_sk2syn(self):
        for synset in wn.all_synsets():
            for lemma in synset.lemmas():
                self.map_sk2syn[lemma.key()] = synset

    @lru_cache()
    def syn2sks(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)
        return list(set([lemma.key() for lemma in synset.lemmas()]))

    @lru_cache()
    def syn2pos(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)
        return synset.pos()

    @lru_cache()
    def syn2lemmas(self, synset, include_pos=False):
        if isinstance(synset, str):
            synset = wn.synset(synset)

        lemmas = synset.lemma_names()
        if include_pos:
            lemmas = ['%s|%s' % (lem, synset.pos()) for lem in lemmas]
        return lemmas

    @lru_cache()
    def syn2lexname(self, synset):
        if isinstance(synset, str):
            synset = wn.synset(synset)
        
        return synset.lexname()

    @lru_cache()
    def syn2offset(self, synset):
        return synset.offset()

    @lru_cache()
    def sk2syn(self, sk):
        return self.map_sk2syn[sk]

    @lru_cache()
    def sk2lemma(self, sk, use_ws=False):
        try:
            lemma_name = wn.lemma_from_key(sk).name()
        except:
            lemma_name = sk.split('%')[0]
        
        if use_ws:
            lemma_name = lemma_name.replace('_', ' ')
        return lemma_name

    @lru_cache()
    def sk2pos(self, sk):
        # merging ADJ with ADJ_SAT
        sk_types_map = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 'a'}
        sk_type = int(sk.split('%')[1].split(':')[0])
        return sk_types_map[sk_type]
        # syn = self.sk2syn(sk)
        # return self.syn2pos(syn)

    @lru_cache()
    def sk2lexname(self, sk):
        syn = self.sk2syn(sk)
        return self.syn2lexname(syn)

    @lru_cache()
    def lemma2syns(self, lemma, pos=None):

        if '|' in lemma:  # custom format, overrides arg
            lemma, pos = lemma.split('|')

        lemma = lemma.replace(' ', '_')

        # merging ADJ with ADJ_SAT
        if pos in ['a', 's']:
            syns = wn.synsets(lemma, pos='a') + wn.synsets(lemma, pos='s')
        else:
            syns = wn.synsets(lemma, pos=pos)

        if len(syns) > 0:
            return syns
        else:
            raise NoSynset('No synset for lemma=\'%s\', pos=\'%s\'.' % (lemma, pos))

    @lru_cache()
    def lemma2sks(self, lemma, pos=None):
        sks = set()

        if '|' in lemma:  # custom format, overrides arg
            lemma, pos = lemma.split('|')
        lemma = lemma.replace(' ', '_')

        # for sk in self.get_all_sks():
        #     if lemma == self.sk2lemma(sk) and pos == self.sk2pos(sk):
        #         sks.add(sk)

        for syn in self.lemma2syns(lemma, pos=pos):
            for sk in self.syn2sks(syn):
                if self.sk2lemma(sk, use_ws=False) == lemma:
                    sks.add(sk)

        return list(sks)

    @lru_cache()
    def lemma2lexnames(self, lemma, pos=None):
        lexnames = set()
        for syn in self.lemma2syns(lemma, pos=pos):
            lexnames.add(self.syn2lexname(syn))
        return list(lexnames)
    
    @lru_cache()
    def synid2syn(self, synid):
        return wn.of2ss(synid)

    @lru_cache()
    def synname2syn(self, synname):
        return wn.synset(synname)

    def get_all_syns(self):
        return list(wn.all_synsets())

    def get_all_lemmas(self, replace_ws=True):
        all_wn_lemmas = list(wn.all_lemma_names())
        if replace_ws:
            all_wn_lemmas = [lemma.replace('_', ' ') for lemma in all_wn_lemmas]
        return all_wn_lemmas

    def get_all_sks(self):
        # return list(self.map_sk2syn.keys())
        return self.map_sk2syn.keys()

    def get_all_lexnames(self):
        lexnames = set()
        for syn in self.get_all_syns():
            lexnames.add(self.syn2lexname(syn))
        return list(lexnames)

    def get_wn_first_sk(self, lemma, postag):
        first_syn = self.lemma2syns(lemma, postag)[0]
        for lem in first_syn.lemmas():
            key = lem.key()
            if key.startswith('{}%'.format(lemma)):
                return key

    def get_syn_antonyms(self, syn):
        syn_antonyms = []
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                syn_antonyms.append(ant.synset())
        return list(set(syn_antonyms))

    def get_sk_antonyms(self, sk):
        syn_antonyms = self.get_syn_antonyms(self.sk2syn(sk))
        sk_antonyms = []
        for syn in syn_antonyms:
            sk_antonyms += self.syn2sks(syn)
        return list(set(sk_antonyms))

    def check_lemma_amb(self, lemma, postag=None):
        return len(self.lemma2syns(lemma, postag)) > 1

    def check_sk_amb(self, sk):
        sk_lemma = self.sk2lemma(sk)
        sk_postag = self.sk2pos(sk)
        return self.check_lemma_amb(sk_lemma, sk_postag)

    def check_sk_1st_sense(self, sk):
        sk_lemma = self.sk2lemma(sk)
        sk_postag = self.sk2pos(sk)
        return sk == self.get_wn_first_sk(sk_lemma, sk_postag)

    def get_all_hypernyms(self, syn, depth=float('inf'), include_self=True):
        if include_self:
            return [syn] + list(syn.closure(lambda s: s.hypernyms(), depth=depth))
        else:
            return list(syn.closure(lambda s: s.hypernyms(), depth=depth))

    def get_all_ambiguous_sks(self):
        ambiguous_sks = []
        for sk in self.get_all_sks():
            if self.check_sk_amb(sk):
                ambiguous_sks.append(sk)
        return ambiguous_sks

    def get_disambiguating_sks(self, lemma, pos):

        lemma_sks = self.lemma2sks(lemma, pos)

        sk_ancestors = {}
        ancestor_counter = Counter()
        for sk in lemma_sks:
            syn = self.sk2syn(sk)

            syn_hypernyms_sks = set()
            for hypernym in self.get_all_hypernyms(syn):
                hypernym_sks = self.syn2sks(hypernym)
                syn_hypernyms_sks.update(hypernym_sks)

            sk_ancestors[sk] = list(syn_hypernyms_sks)
            ancestor_counter.update(sk_ancestors[sk])

        # keep only unique ancestors
        for sk, ancestors in sk_ancestors.items():
            sk_ancestors[sk] = [sk_ for sk_ in ancestors if ancestor_counter[sk_] == 1]

        # invert ancestors to lemma sks
        disambiguating_sks = {}
        for sk, ancestors in sk_ancestors.items():
            for sk_ in ancestors:
                disambiguating_sks[sk_] = sk
        
        for sk in lemma_sks:
            disambiguating_sks[sk] = sk

        return disambiguating_sks

    def convert_postag(self, postag):
        # merges ADJ with ADJ_SAT
        postags_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r', 'ADJ_SAT': 'a'}
        if postag in postags_map.values():
            return postag
        elif postag in postags_map:
            return postags_map[postag]
        else:
            # raise exception
            return None

    def wup_similarity(self, syn1, syn2):
        return syn1.wup_similarity(syn2)

if __name__ == '__main__':

    wn_utils = WN_Utils()
    print(wn_utils.lemma2sks('hydrophobia'))
