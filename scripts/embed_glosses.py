import time
import argparse
import logging
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

import sys, os  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def wn_synset2keys(synset):
    if isinstance(synset, str):
        synset = wn.synset(synset)
    return list(set([lemma.key() for lemma in synset.lemmas()]))


def fix_lemma(lemma):
    return lemma.replace('_', ' ')


def get_sense_data():
    data = []

    for synset in wn.all_synsets():
        all_lemmas = [fix_lemma(lemma.name()) for lemma in synset.lemmas()]
        gloss = ' '.join(word_tokenize(synset.definition()))
        for lemma in synset.lemmas():
            lemma_name = fix_lemma(lemma.name())
            d_str = lemma_name + ' - ' + ' , '.join(all_lemmas) + ' - ' + gloss
            data.append((synset, lemma.key(), d_str))

    data = sorted(data, key=lambda x: x[0])
    return data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Creates sense embeddings based on glosses and lemmas.')
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-large-cased')
    parser.add_argument('-sense_level', type=str, default='sensekey', help='Representation Level', required=False, choices=['synset', 'sensekey'])
    parser.add_argument('-subword_op', type=str, default='mean', help='Subword Reconstruction Strategy', required=False, choices=['mean', 'first', 'sum'])
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False, choices=['mean', 'sum', 'concat', 'ws'])
    parser.add_argument('-weights_path', type=str, default='', help='Path to layer weights', required=False)
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
    args = parser.parse_args()

    if args.layer_op == 'ws' and args.weights_path == '':
        raise(BaseException('Weights path must be given with layer_op \'ws\''))

    if args.layer_op == 'ws':
        args.layers = 'all'  # override

    if args.layers == 'all':
        if '-base' in args.nlm_id or args.nlm_id == 'albert-xxlarge-v2':
            nmax_layers = 12 + 1
        else:
            nmax_layers = 24 + 1
        args.layers = [-n for n in range(1, nmax_layers + 1)]
    else:
        args.layers = [int(n) for n in args.layers.split(' ')]

    encoder_cfg = {
        'model_name_or_path': args.nlm_id,
        'min_seq_len': 0,
        'max_seq_len': args.max_seq_len,
        'layers': args.layers,
        'layer_op': args.layer_op,
        'weights_path': args.weights_path,
        'subword_op': args.subword_op
    }

    if encoder_cfg['model_name_or_path'].split('-')[0] in ['roberta', 'xlmr']:
        encoder = FairSeqEncoder(encoder_cfg)
    else:
        encoder = TransformersEncoder(encoder_cfg)


    logging.info('Preparing Gloss Data ...')
    glosses = get_sense_data()
    glosses_vecs = defaultdict(list)

    logging.info('Embedding Senses ...')
    t0 = time.time()
    for batch_idx, glosses_batch in enumerate(chunks(glosses, args.batch_size)):
        dfns = [e[-1] for e in glosses_batch]

        batch_tok_vecs = encoder.token_embeddings([dfn.split() for dfn in dfns])

        batch_dfn_vecs = []
        for dfn_tok_vecs in batch_tok_vecs:
            dfn_vec = np.mean([vec for tok, vec in dfn_tok_vecs], axis=0)
            batch_dfn_vecs.append(dfn_vec)

        for (synset, sensekey, dfn), dfn_vec in zip(glosses_batch, batch_dfn_vecs):
            glosses_vecs[sensekey].append(dfn_vec)

        t_span = time.time() - t0
        n = (batch_idx + 1) * args.batch_size
        logging.info('%d/%d at %.3f per sec' % (n, len(glosses), n/t_span))


    logging.info('Writing Vectors %s ...' % args.out_path)
    with open(args.out_path, 'w') as vecs_senses_f:
        for sensekey, sensekey_vecs in glosses_vecs.items():
            vec = np.array(sensekey_vecs[0])
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            vecs_senses_f.write('%s %s\n' % (sensekey, vec_str))
    logging.info('Done')
