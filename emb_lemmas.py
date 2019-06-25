import logging
import argparse
from nltk.corpus import wordnet as wn
import fastText


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def get_senselemma(sensekey):  # replicating method used in SenseVSM
    return sensekey.split('%')[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Creates static word embeddings for WordNet synsets (lemmas only).')
    parser.add_argument('-ft_path', help='Path to fastText vectors', required=False,
                        default='external/fasttext/crawl-300d-2M-subword.bin')
    parser.add_argument('-out_path', help='Path to resulting lemma vectors', required=True)
    args = parser.parse_args()

    logging.info('Loading fastText model ...')
    model = fastText.load_model(args.ft_path)

    logging.info('Creating lemma embeddings ...')
    sensekey_vecs = []
    for synset_idx, synset in enumerate(wn.all_synsets()):
        for lemma in synset.lemmas():
            sensekey = lemma.key()
            sensekey_lemma = get_senselemma(sensekey)
            lemma_vec = model.get_word_vector(sensekey_lemma)
            sensekey_vecs.append((sensekey, lemma_vec))

        if synset_idx % 10000 == 0:
            logging.info('at synset %d' % synset_idx)

    sensekey_vecs = sorted(sensekey_vecs, key=lambda x: x[0])

    logging.info('Writing lemma embeddings ...')
    with open(args.out_path, 'w') as vecs_f:
        for sensekey, vec in sensekey_vecs:
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            vecs_f.write('%s %s\n' % (sensekey, vec_str))

    logging.info('Done')
