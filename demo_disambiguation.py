import numpy as np
from transformers_encoder import TransformersEncoder
from vectorspace import SensesVSM

import spacy
en_nlp = spacy.load('en_core_web_sm')  # required for lemmatization and POS-tagging

from wn_utils import WN_Utils
wn_utils = WN_Utils()  # WordNet auxilliary methods (just for describing results)


# NLM/LMMS paths and parameters
vecs_path = '/media/dan/ElementsWORK/albert-xxlarge-v2/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt'
wsd_encoder_cfg = {
    'model_name_or_path': 'albert-xxlarge-v2',
    'min_seq_len': 0,
    'max_seq_len': 512,
    'layers': [-n for n in range(1, 12 + 1)],  # all layers, with reversed indices
    'layer_op': 'ws',
    'weights_path': 'data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt',
    'subword_op': 'mean'
}


print('Loading NLM and sense embeddings ...')  # (takes a while)
wsd_encoder = TransformersEncoder(wsd_encoder_cfg)
senses_vsm = SensesVSM(vecs_path, normalize=True)
print('Done')


# input sentence, with indices of token/span to disambiguate
sentence = 'This mouse has no batteries.'
target_idxs = [1]  # for 'mouse'


# use spacy to automatically determine lemma and POS (replace with your favorite NLP toolkit)
doc = en_nlp(sentence)
target_lemma = '_'.join([doc[i].lemma_ for i in target_idxs])
target_pos = doc[target_idxs[0]].pos_


# retrieve contextual embedding for target token/span
tokens = [t.text for t in doc]
ctx_embeddings = wsd_encoder.token_embeddings([tokens])[0]
target_embedding = np.array([ctx_embeddings[i][1] for i in target_idxs]).mean(axis=0)
target_embedding = target_embedding / np.linalg.norm(target_embedding)


# find sense embeddings that are nearest-neighbors to the target contextual embedding
# candidates restricted by lemma and part-of-speech
matches = senses_vsm.match_senses(target_embedding, lemma=target_lemma, postag=target_pos, topn=5)


# report matches, showing also additional info from WordNet for each match
for sk, sim in matches:
    syn = wn_utils.sk2syn(sk)
    lex = wn_utils.sk2lexname(sk)
    dfn = syn.definition()
    print('%f - %s (%s; %s): %s' % (sim, sk, syn.name(), lex, dfn))

# should output:
# 0.594206 - mouse%1:06:00:: (mouse.n.04): a hand-operated electronic device ...
# 0.552628 - mouse%1:05:00:: (mouse.n.01): any of numerous small rodents typically ...
# 0.332615 - mouse%1:18:00:: (mouse.n.03): person who is quiet or timid
# 0.234823 - mouse%1:26:00:: (shiner.n.01): a swollen bruise caused by a blow to the eye
