import logging
import argparse
from time import time

import lxml.etree
import numpy as np

from bert_as_service import bert_embed
from bert_as_service import tokenizer as bert_tokenizer


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def get_sense_mapping(eval_path):
    sensekey_mapping = {}
    with open(eval_path) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            sensekey_mapping[id_] = keys
    return sensekey_mapping


def read_xml_sents(xml_path):
    with open(xml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('<sentence '):
                sent_elems = [line]
            elif line.startswith('<wf ') or line.startswith('<instance '):
                sent_elems.append(line)
            elif line.startswith('</sentence>'):
                sent_elems.append(line)
                yield lxml.etree.fromstring(''.join(sent_elems))


def train(train_path, eval_path, vecs_path, merge_strategy='mean', max_seq_len=512, max_instances=float('inf')):

    sense_vecs = {}
    sense_mapping = get_sense_mapping(eval_path)

    batch, batch_idx, batch_t0 = [], 0, time()
    for sent_idx, sent_et in enumerate(read_xml_sents(train_path)):
        entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses', 'pos', 'id']}
        for ch in sent_et.getchildren():
            for k, v in ch.items():
                entry[k].append(v)
            entry['token_mw'].append(ch.text)

            if 'id' in ch.attrib.keys():
                entry['senses'].append(sense_mapping[ch.attrib['id']])
            else:
                entry['senses'].append(None)

        entry['token'] = sum([t.split() for t in entry['token_mw']], [])
        entry['sentence'] = ' '.join([t for t in entry['token_mw']])

        bert_tokens = bert_tokenizer.tokenize(entry['sentence'])
        if len(bert_tokens) < max_seq_len:
            batch.append(entry)

        if len(batch) == args.batch_size:

            batch_sents = [e['sentence'] for e in batch]
            batch_bert = bert_embed(batch_sents, merge_strategy=merge_strategy)

            for sent_info, sent_bert in zip(batch, batch_bert):
                # handling multi-word expressions, mapping allows matching tokens with mw features
                idx_map_abs = []
                idx_map_rel = [(i, list(range(len(t.split()))))
                                for i, t in enumerate(sent_info['token_mw'])]
                token_counter = 0
                for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                    idx_tokens = [i+token_counter for i in idx_tokens]
                    token_counter += len(idx_tokens)
                    idx_map_abs.append([idx_group, idx_tokens])

                for mw_idx, tok_idxs in idx_map_abs:
                    if sent_info['senses'][mw_idx] is None:
                        continue

                    vec = np.array([sent_bert[i][1] for i in tok_idxs], dtype=np.float32).mean(axis=0)

                    for sense in sent_info['senses'][mw_idx]:
                        try:
                            if sense_vecs[sense]['vecs_num'] < max_instances:
                                sense_vecs[sense]['vecs_sum'] += vec
                                sense_vecs[sense]['vecs_num'] += 1
                        except KeyError:
                            sense_vecs[sense] = {'vecs_sum': vec, 'vecs_num': 1}

            batch_tspan = time() - batch_t0
            logging.info('%.3f sents/sec - %d sents, %d senses' % (args.batch_size/batch_tspan, sent_idx, len(sense_vecs)))

            batch, batch_t0 = [], time()
            batch_idx += 1

    logging.info('#sents: %d' % sent_idx)

    logging.info('Writing Sense Vectors ...')
    with open(vecs_path, 'w') as vecs_f:
        for sense, vecs_info in sense_vecs.items():
            vec = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            vecs_f.write('%s %s\n' % (sense, vec_str))
    logging.info('Written %s' % vecs_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Initial Sense Embeddings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-dataset', default='semcor', help='Name of dataset', required=False,
                        choices=['semcor', 'semcor_omsti'])
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-max_instances', type=float, default=float('inf'), help='Maximum number of examples for each sense', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
    args = parser.parse_args()

    layers = list(range(-4, 0))[::-1]
    layers_str = '%d_%d' % (layers[0], layers[-1])

    if args.dataset == 'semcor':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
        keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
    elif args.dataset == 'semcor_omsti':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
        keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'

    train(train_path, keys_path, args.out_path, args.merge_strategy, args.max_seq_len, args.max_instances)
