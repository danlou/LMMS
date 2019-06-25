"""
Alternative to concat.py that we tested with.
"""
import logging
import argparse
import numpy as np
from datetime import datetime
from orderedset import OrderedSet


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def normalize(vec_as_list):
    vec = np.array(vec_as_list)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def avg(v1_path, v2_path, out_path, norm=True):

    logging.info('Loading %s ...' % v1_path)  # i.e. fasttext tokens
    txt1_vecs = {}
    with open(v1_path) as txt1_f:
        for line in txt1_f:
            info = line.split()
            label, vec_str = info[0], info[1:]
            vec = [float(v) for v in vec_str]
            if norm:
                vec = normalize(vec)
            txt1_vecs[label] = vec

    logging.info('Loading %s ...' % v2_path)  # i.e. BERT sentences
    txt2_vecs = {}
    with open(v2_path) as txt2_f:
        for line in txt2_f:
            info = line.split()
            label, vec_str = info[0], info[1:]
            vec = [float(v) for v in vec_str]
            if norm:
                vec = normalize(vec)
            txt2_vecs[label] = vec

    logging.info('Combining vecs (avg) ...')
    txt1_labels = OrderedSet(txt1_vecs.keys())  # first sets the order
    for label1 in txt1_labels:
        v1 = txt1_vecs[label1]
        v2 = txt2_vecs[label1]

        assert len(v1) == len(v2)
        txt1_vecs[label1] = [(v1_ + v2_) / 2 for v1_, v2_ in zip(v1, v2)]

    logging.info('Writing %s ...' % out_path)
    with open(out_path, 'w') as merged_f:
        for label in txt1_labels:
            vec = txt1_vecs[label]
            vec_str = [str(round(v, 6)) for v in vec]
            merged_f.write('%s %s\n' % (label, ' '.join(vec_str)))

    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenates and normalized vector .txt files.')
    parser.add_argument('-v1_path', help='Path to vector set 1', required=True)
    parser.add_argument('-v2_path', help='Path to vector set 2', required=True)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
    args = parser.parse_args()

    avg(args.v1_path, args.v2_path, args.out_path)
