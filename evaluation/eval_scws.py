"""
LMMS-SP solution for Stanford's Contextual Word Similarities (SCWS)

This script does not have a CLI.
"""

import os
import sys
from datetime import datetime

from scipy.stats import spearmanr
import numpy as np

import sys  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder

# from vectorspace import VSM
from vectorspace import SensesVSM


def load_scws(path='external/scws/ratings.txt', restr_pos_pair=None):
    
    def unbolded(context):
        context = str.replace(context, '<b>', '')
        return str.replace(context, '</b>', '')
    
    def get_word_idx(tokens, word):
        try:
            return tokens.index(word)
        except ValueError:
            tokens = [t.lower() for t in tokens]
            return tokens.index(word.lower())

    instances = []
    ratings = []

    with open(path) as f:
        for line in f:
            elems = line.strip().split('\t')
            inst_id, word1, word1_pos, word2, word2_pos, ctx1, ctx2 = elems[:7]
            avg_rating, all_ratings = float(elems[7]), elems[8:]

            # restrict to specific pairs of POS tags (unordered)
            if restr_pos_pair != None:
                if set([word1_pos, word2_pos]) != set(restr_pos_pair):
                    continue

            ctx1 = unbolded(ctx1)
            ctx2 = unbolded(ctx2)
            ctx1_tokens = ctx1.split(' ')
            ctx2_tokens = ctx2.split(' ')
            word1_idx = get_word_idx(ctx1_tokens, word1)
            word2_idx = get_word_idx(ctx2_tokens, word2)

            e = {'ctx1_tokens': ctx1_tokens, 'ctx2_tokens': ctx2_tokens,
                 'word1_idx': word1_idx, 'word2_idx': word2_idx,
                 'word1_pos': word1_pos, 'word2_pos': word2_pos}
            instances.append(e)
            ratings.append(avg_rating)

    return instances, ratings



if __name__ == '__main__':

    vectors_path = 'data/vectors/' 

    # # BERT-L
    # encoder_cfg = {
    #     'model_name_or_path': 'bert-large-cased',
    #     'min_seq_len': 0,
    #     'max_seq_len': 256,
    #     'layers': [-n for n in range(1, 24 + 1)],
    #     'layer_op': 'ws',
    #     'weights_path': 'data/weights/lmms-sp-wsd.bert-large-cased.weights.txt',
    #     'subword_op': 'mean'
    # }
    # sensevsm = SensesVSM(vectors_path + 'bert-large-cased/lmms-sp-wsd.bert-large-cased.vectors.txt', normalize=True)

    # # XLNET-L
    # encoder_cfg = {
    #     'model_name_or_path': 'xlnet-large-cased',
    #     'min_seq_len': 0,
    #     'max_seq_len': 256,
    #     'layers': [-n for n in range(1, 24 + 1)],
    #     'layer_op': 'ws',
    #     'weights_path': 'data/weights/lmms-sp-wsd.xlnet-large-cased.weights.txt',
    #     'subword_op': 'mean'
    # }
    # sensevsm = SensesVSM(vectors_path + 'xlnet-large-cased/lmms-sp-wsd.xlnet-large-cased.vectors.txt', normalize=True)

    # # RoBERTa-L
    # encoder_cfg = {
    #     'model_name_or_path': 'roberta-large',
    #     'min_seq_len': 0,
    #     'max_seq_len': 256,
    #     'layers': [-n for n in range(1, 24 + 1)],
    #     'layer_op': 'ws',
    #     'weights_path': 'data/weights/lmms-sp-wsd.roberta-large.weights.txt',
    #     'subword_op': 'mean'
    # }
    # sensevsm = SensesVSM(vectors_path + 'roberta-large/lmms-sp-wsd.roberta-large.vectors.txt', normalize=True)

    # ALBERT-XXL
    encoder_cfg = {
        'model_name_or_path': 'albert-xxlarge-v2',
        'min_seq_len': 0,
        'max_seq_len': 256,
        'layers': [-n for n in range(1, 12 + 1)],
        'layer_op': 'ws',
        'weights_path': 'data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt',
        'subword_op': 'mean'
    }
    sensevsm = SensesVSM(vectors_path + 'albert-xxlarge-v2/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt', normalize=True)

    if encoder_cfg['model_name_or_path'].split('-')[0] in ['roberta', 'xlmr']:
        encoder = FairSeqEncoder(encoder_cfg)
    else:
        encoder = TransformersEncoder(encoder_cfg)

    instances, ratings = load_scws()
    # instances, ratings = load_scws(restr_pos_pair={'a', 'a'})

    ctx_sims, sk_sims = [], []
    for inst_idx, inst in enumerate(instances):
        print(datetime.now(), '%d/%d' % (inst_idx, len(instances)))

        ctx1_embeddings = encoder.token_embeddings([inst['ctx1_tokens']])[0]
        ctx2_embeddings = encoder.token_embeddings([inst['ctx2_tokens']])[0]

        w1_ctx1_embedding = ctx1_embeddings[inst['word1_idx']][1]
        w2_ctx2_embedding = ctx2_embeddings[inst['word2_idx']][1]

        w1_ctx1_embedding = w1_ctx1_embedding / np.linalg.norm(w1_ctx1_embedding)
        w2_ctx2_embedding = w2_ctx2_embedding / np.linalg.norm(w2_ctx2_embedding)

        ctx_sim = np.dot(w1_ctx1_embedding, w2_ctx2_embedding).tolist()
        ctx_sims.append(ctx_sim)

        # Match LMMS embeddings
        w1_ctx1_matches = sensevsm.match_senses(w1_ctx1_embedding, None, None, topn=None)
        w2_ctx2_matches = sensevsm.match_senses(w2_ctx2_embedding, None, None, topn=None)

        sk1_ctx1_embedding = sensevsm.get_vec(w1_ctx1_matches[0][0])
        sk2_ctx2_embedding = sensevsm.get_vec(w2_ctx2_matches[0][0])

        sk_sim = np.dot(sk1_ctx1_embedding, sk2_ctx2_embedding).tolist()
        sk_sims.append(sk_sim)

    final_sims = [(ctx_sim + sk_sim) / 2 for ctx_sim, sk_sim in zip(ctx_sims, sk_sims)]
    print('CTX:', spearmanr(ratings, ctx_sims))
    print('SNS:', spearmanr(ratings, sk_sims))
    print('MIX:', spearmanr(ratings, final_sims))
