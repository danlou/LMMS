import time
import argparse
import logging
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

from bert_as_service import bert_embed_sents
from bert_as_service import bert_embed


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
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
    args = parser.parse_args()

    pooling_strategy = 'REDUCE_MEAN' # important parameter to replicate results using bert-as-service
    # merge_strategy = 'mean'  # just for NONE

    logging.info('Preparing Gloss Data ...')
    glosses = get_sense_data()
    glosses_vecs = defaultdict(list)

    logging.info('Embedding Senses ...')
    t0 = time.time()
    for batch_idx, glosses_batch in enumerate(chunks(glosses, args.batch_size)):
        dfns = [e[-1] for e in glosses_batch]

        if pooling_strategy == 'REDUCE_MEAN':
            dfns_bert = bert_embed_sents(dfns, strategy=pooling_strategy)
        # elif pooling_strategy == 'NONE':
        #     dfns_bert = bert_embed(dfns, merge_strategy=merge_strategy)
        #     # to-do ...

        for (synset, sensekey, dfn), dfn_bert in zip(glosses_batch, dfns_bert):
            assert dfn_bert[0] == dfn
            dfn_vec = dfn_bert[1]
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
