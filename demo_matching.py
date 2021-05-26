import numpy as np
from transformers_encoder import TransformersEncoder
from vectorspace import SensesVSM

from wn_utils import WN_Utils
wn_utils = WN_Utils()  # WordNet auxilliary methods (just for describing results)


# NLM/LMMS paths and parameters
vecs_path = '/media/dan/ElementsWORK/albert-xxlarge-v2/lmms-sp-usm.albert-xxlarge-v2.vectors.txt'
usm_encoder_cfg = {
    'model_name_or_path': 'albert-xxlarge-v2',
    'min_seq_len': 0,
    'max_seq_len': 512,
    'layers': [-n for n in range(1, 12 + 1)],  # all layers, with reversed indices
    'layer_op': 'ws',
    'weights_path': 'data/weights/lmms-sp-usm.albert-xxlarge-v2.weights.txt',
    'subword_op': 'mean'
}


print('Loading NLM and sense embeddings ...')  # (takes a while)
usm_encoder = TransformersEncoder(usm_encoder_cfg)
senses_vsm = SensesVSM(vecs_path, normalize=True)
print('Done')


# input sentence, with indices of token/span to disambiguate
sentence = 'Marlon Brando played Corleone in Godfather'  # assuming pre-tokenized (whitespace) for simplicity
target_idxs = [3]  # for 'Corleone', or [0, 1] for 'Marlon Brando'


# retrieve contextual embedding for target token/span
tokens = sentence.split()
ctx_embeddings = usm_encoder.token_embeddings([tokens])[0]
target_embedding = np.array([ctx_embeddings[i][1] for i in target_idxs]).mean(axis=0)
target_embedding = target_embedding / np.linalg.norm(target_embedding)


# find sense embeddings that are nearest-neighbors to the target contextual embedding
matches = senses_vsm.match_senses(target_embedding, topn=5)


# report matches, showing also additional info from WordNet for each match
for sk, sim in matches:
    syn = wn_utils.sk2syn(sk)
    lex = wn_utils.sk2lexname(sk)
    dfn = syn.definition()
    print('%f - %s (%s; %s): %s' % (sim, sk, syn.name(), lex, dfn))


# for 'Corleone', should output:
# 0.620031 - capone%1:18:00:: (capone.n.01; noun.person): United States gangster who terrorized Chicago ...
# 0.590865 - alphonse_capone%1:18:00:: (capone.n.01; noun.person): United States gangster who terrorized ...
# 0.585405 - al_capone%1:18:00:: (capone.n.01; noun.person): United States gangster who terrorized Chicago ...
# 0.564465 - veronese%1:18:00:: (veronese.n.01; noun.person): Italian painter of the Venetian school (1528-1588)
# 0.557574 - corday%1:18:00:: (corday.n.01; noun.person): French revolutionary heroine (a Girondist) who assassinated ...
# 
# for 'Marlon Brando', should output:
# 0.545564 - jimmy_cagney%1:18:00:: (cagney.n.01; noun.person): United States film actor known for his portrayals of tough ...
# 0.545254 - boris_karloff%1:18:00:: (karloff.n.01; noun.person): United States film actor (born in England) noted for his ...
# 0.543891 - robert_de_niro%1:18:00:: (de_niro.n.01; noun.person): United States film actor who frequently plays tough ...
# 0.539469 - henry_fonda%1:18:00:: (fonda.n.02; noun.person): United States film actor (1905-1982)
# 0.537894 - fonda%1:18:00:: (fonda.n.02; noun.person): United States film actor (1905-1982)
