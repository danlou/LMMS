"""
LMMS-SP solution for Graded Word Similarity in Context (GWCS), SemEval 2020 - Task 3
https://competitions.codalab.org/competitions/20905

Evaluation script based on the task provided baseline.

Requires spacy for tokenization.

This script does not have a CLI.
"""

import os
import csv
import sys
from unidecode import unidecode

import numpy as np
import spacy
en_nlp = spacy.load('en_core_web_sm')

import sys  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder

# from vectorspace import VSM
from vectorspace import SensesVSM


# mode = 'practice'
mode = 'final'

if mode == 'practice':
    task_langs = ['en']
    basepath_data = 'external/gwcs/practice_kit_final/data/'
    basepath_results = 'results/practice_'

elif mode == 'final':
    task_langs = ['en']
    basepath_data = 'external/gwcs/evaluation_kit_final/data/'
    basepath_results = 'results/final_'


# Testing ALBERT-XXL
en_encoder_cfg = {
    'model_name_or_path': 'albert-xxlarge-v2',
    'min_seq_len': 0,
    'max_seq_len': 256,
    'layers': [-n for n in range(1, 12 + 1)],
    'layer_op': 'ws',
    'weights_path': 'data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt',
    'subword_op': 'mean'
}
en_sensevsm = SensesVSM('data/vectors/albert-xxlarge-v2/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt', normalize=True)


if en_encoder_cfg['model_name_or_path'].split('-')[0] in ['roberta', 'xlmr']:
    en_encoder = FairSeqEncoder(en_encoder_cfg)
else:
    en_encoder = TransformersEncoder(en_encoder_cfg)


def unbolded(context):
    context = str.replace(context, '<strong>', '')
    return str.replace(context, '</strong>', '')


def tokenize(context, remove_bold=True, lang='en'):

    if remove_bold:
        context = unbolded(context)

    context = ' '.join(context.split())  # normalize spacing
    tokens = [tok.text for tok in en_nlp(context)]
    tokens = [unidecode(tok) for tok in tokens]

    return tokens


def get_word_embedding(ctx_embeddings, tgt_word):
    for (tok, emb) in ctx_embeddings:
        # if unidecode(tok) == unidecode(row['word1_context1'].lower()):
        if tok == tgt_word:
            return emb
    
    # try a hotfix on tokenization
    for (tok, emb) in ctx_embeddings:
        if tgt_word in tok.split('-'):
            return emb



for lang in task_langs:

    similarities = []

    print(f'\nLANGUAGE: {lang.upper()}\n')

    with open(basepath_data + f'data_{lang}.tsv', 'r') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for index, row in enumerate(csvreader):

            print(f"{index} {row['word1']}-{row['word2']}")

            ### CONTEXT1 ###
            print('CTX1', row['context1'])
            print(f"{index} {row['word1_context1']}-{row['word2_context1']}")

            ctx1_tokens = tokenize(row['context1'], lang=lang)
            # print(ctx1_tokens)

            ctx1_embeddings = en_encoder.token_embeddings([ctx1_tokens])[0]
            w1_ctx1 = unidecode(row['word1_context1'])
            w2_ctx1 = unidecode(row['word2_context1'])
    
            w1_ctx1_embedding = get_word_embedding(ctx1_embeddings, w1_ctx1)
            w2_ctx1_embedding = get_word_embedding(ctx1_embeddings, w2_ctx1)

            w1_ctx1_embedding = w1_ctx1_embedding / np.linalg.norm(w1_ctx1_embedding)
            w2_ctx1_embedding = w2_ctx1_embedding / np.linalg.norm(w2_ctx1_embedding)
            ctx1_sim_embs = np.dot(w1_ctx1_embedding, w2_ctx1_embedding).tolist()

            print(w1_ctx1, w2_ctx1)
            print('ctx1_sim_embs:', ctx1_sim_embs)

            if en_sensevsm.ndims in [1024+1024, 4096+4096]:  # if using concatenated gloss embeddings
                w1_ctx1_embedding = np.hstack((w1_ctx1_embedding, w1_ctx1_embedding))
                w2_ctx1_embedding = np.hstack((w2_ctx1_embedding, w2_ctx1_embedding))
                w1_ctx1_embedding = w1_ctx1_embedding / np.linalg.norm(w1_ctx1_embedding)
                w2_ctx1_embedding = w2_ctx1_embedding / np.linalg.norm(w2_ctx1_embedding)

            w1_ctx1_matches = en_sensevsm.match_senses(w1_ctx1_embedding, None, None, topn=None)
            w2_ctx1_matches = en_sensevsm.match_senses(w2_ctx1_embedding, None, None, topn=None)

            sk1_ctx1_embedding = en_sensevsm.get_vec(w1_ctx1_matches[0][0])
            sk2_ctx1_embedding = en_sensevsm.get_vec(w2_ctx1_matches[0][0])
            ctx1_sim_sks = np.dot(sk1_ctx1_embedding, sk2_ctx1_embedding).tolist()

            print('SIMS:', ctx1_sim_embs, ctx1_sim_sks)
            sim_ctx1 = (ctx1_sim_embs + ctx1_sim_sks) / 2

            ### CONTEXT2 ###
            print('CTX2', row['context2'])
            print(f"{index} {row['word1_context2']}-{row['word2_context2']}")

            ctx2_tokens = tokenize(row['context2'], lang=lang)
            # print(ctx2_tokens)

            ctx2_embeddings = en_encoder.token_embeddings([ctx2_tokens])[0]
            w1_ctx2 = unidecode(row['word1_context2'])
            w2_ctx2 = unidecode(row['word2_context2'])

            w1_ctx2_embedding = get_word_embedding(ctx2_embeddings, w1_ctx2)
            w2_ctx2_embedding = get_word_embedding(ctx2_embeddings, w2_ctx2)

            w1_ctx2_embedding = w1_ctx2_embedding / np.linalg.norm(w1_ctx2_embedding)
            w2_ctx2_embedding = w2_ctx2_embedding / np.linalg.norm(w2_ctx2_embedding)
            ctx2_sim_embs = np.dot(w1_ctx2_embedding, w2_ctx2_embedding).tolist()

            print(w1_ctx2, w2_ctx2)
            print('ctx2_sim_embs:', ctx2_sim_embs)

            if en_sensevsm.ndims in [1024+1024, 4096+4096]:  # if using concatenated gloss embeddings
                w1_ctx2_embedding = np.hstack((w1_ctx2_embedding, w1_ctx2_embedding))
                w2_ctx2_embedding = np.hstack((w2_ctx2_embedding, w2_ctx2_embedding))
                w1_ctx2_embedding = w1_ctx2_embedding / np.linalg.norm(w1_ctx2_embedding)
                w2_ctx2_embedding = w2_ctx2_embedding / np.linalg.norm(w2_ctx2_embedding)

            w1_ctx2_matches = en_sensevsm.match_senses(w1_ctx2_embedding, None, None, topn=None)
            w2_ctx2_matches = en_sensevsm.match_senses(w2_ctx2_embedding, None, None, topn=None)

            sk1_ctx2_embedding = en_sensevsm.get_vec(w1_ctx2_matches[0][0])
            sk2_ctx2_embedding = en_sensevsm.get_vec(w2_ctx2_matches[0][0])
            ctx2_sim_sks = np.dot(sk1_ctx2_embedding, sk2_ctx2_embedding).tolist()

            print('SIMS:', ctx2_sim_embs, ctx2_sim_sks)
            sim_ctx2 = (ctx2_sim_embs + ctx2_sim_sks) / 2

            ###
            print([sim_ctx1, sim_ctx2])
            similarities.append([sim_ctx1, sim_ctx2])
            print()

    columns = ['change']
    with open(basepath_results + f'se20_subtask1_{lang}.tsv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
        csvwriter.writerow(columns)
        for sim in similarities:
            change = sim[1] - sim[0]
            csvwriter.writerow([change])

    columns = ['sim_context1', 'sim_context2']
    with open(basepath_results + f'se20_subtask2_{lang}.tsv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='', escapechar='~')
        csvwriter.writerow(columns)
        for sim in similarities:
            csvwriter.writerow(sim)
