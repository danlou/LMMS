import logging
import argparse
from time import time

import lxml.etree
import numpy as np
from nltk.corpus import wordnet as wn

import sys, os  # for parent directory imports
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from transformers_encoder import TransformersEncoder
from fairseq_encoder import FairSeqEncoder


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
    """Yield successive n-sized chunks from given list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line.startswith('<sentence ') or line == '<sentence>':
                sent_elems = [line]
            elif line.startswith('<wf ') or line.startswith('<instance '):
                sent_elems.append(line)
            elif line.startswith('</sentence>'):
                sent_elems.append(line)
                try:
                    yield lxml.etree.fromstring(''.join(sent_elems))
                except lxml.etree.XMLSyntaxError:
                    logging.fatal('XML Parsing Error: %d' % line_idx)
                    input('...')


def process_et_sent(sent_et, sense_mapping):

    entry = {f: [] for f in ['token_mw', 'senses']}
    for ch in sent_et.getchildren():
        for k, v in ch.items():
            if k in {'token_mw', 'senses'}:
                entry[k].append(v)
        
        if (ch.text is not None) and len(ch.text) < 32:
            entry['token_mw'].append(ch.text)
        else:
            entry['token_mw'].append('UNK')

        if 'id' in ch.attrib.keys():
            entry['senses'].append(sense_mapping[ch.attrib['id']])
        else:
            entry['senses'].append(None)

    # handling multi-word expressions, mapping allows matching tokens with mw features
    idx_map_abs = []
    idx_map_rel = [(i, list(range(len(t.split())))) for i, t in enumerate(entry['token_mw'])]
    token_counter = 0
    for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
        idx_tokens = [i+token_counter for i in idx_tokens]
        token_counter += len(idx_tokens)
        idx_map_abs.append([idx_group, idx_tokens])
    entry['idx_map'] = idx_map_abs
    entry['n_toks'] = token_counter

    return entry


def gen_vecs(args, encoder, train_path, eval_path):

    sense_vecs = {}
    sense_mapping = get_sense_mapping(eval_path)

    logging.info('Preparing docs ...')
    docs = []
    for sent_idx, sent_et in enumerate(read_xml_sents(train_path)):
        entry = process_et_sent(sent_et, sense_mapping)
        docs.append(entry)

        # if sent_idx % 100000 == 0:
        #     logging.info('sent_idx: %d' % sent_idx)

    docs = sorted(docs, key=lambda x: x['n_toks'])

    logging.info('Processing docs ...')
    sent_idx = 0
    n_failed = 0
    for batch in chunks(docs, args.batch_size):
        batch_t0 = time()

        batch_sent_toks_mw = [e['token_mw'] for e in batch]
        batch_sent_toks = [sum([t.split() for t in toks_mw], []) for toks_mw in batch_sent_toks_mw]
        batch_embs = encoder.token_embeddings(batch_sent_toks)

        for sent_info, sent_embs in zip(batch, batch_embs):
            sent_idx += 1

            for mw_idx, tok_idxs in sent_info['idx_map']:
                if sent_info['senses'][mw_idx] is None:
                    continue

                vec = np.array([sent_embs[i][1] for i in tok_idxs], dtype=np.float64).mean(axis=0)

                for sk in sent_info['senses'][mw_idx]:

                    if args.sense_level == 'sensekey':
                        sense_id = sk
                    elif args.sense_level == 'synset':
                        sense_id = map_sk2syn[sk]

                    if sense_id not in sense_vecs:
                        sense_vecs[sense_id] = {'n': 1, 'vec': vec}
                    
                    elif len(sense_vecs[sense_id]) < args.max_instances:
                        sense_vecs[sense_id]['n'] += 1
                        sense_vecs[sense_id]['vec'] += vec

            batch_tspan = time() - batch_t0
            progress = sent_idx/len(docs) * 100
            logging.info('PROGRESS: %.3f - %.3f sents/sec - %d/%d sents, %d sks' % (progress, args.batch_size/batch_tspan, sent_idx, len(docs), len(sense_vecs)))

    logging.info('#sents final: %d' % sent_idx)
    logging.info('#vecs: %d' % len(sense_vecs))
    logging.info('#failed batches: %d' % n_failed)

    logging.info('Writing sense vecs to %s ...' % args.out_path)
    with open(args.out_path, 'w') as vecs_f:
        for sense_id, vec_info in sense_vecs.items():
            vec = vec_info['vec'] / vec_info['n']
            vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
            vecs_f.write('%s %s\n' % (sense_id, vec_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create sense embeddings from annotated corpora.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nlm_id', help='HF Transfomers model name', required=False, default='bert-large-cased')
    parser.add_argument('-sense_level', type=str, default='sensekey', help='Representation Level', required=False, choices=['synset', 'sensekey'])
    parser.add_argument('-weights_path', type=str, default='', help='Path to layer weights', required=False)
    parser.add_argument('-eval_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-dataset', default='semcor', help='Name of dataset', required=True, choices=['semcor', 'semcor_uwa10'])
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length', required=False)
    parser.add_argument('-subword_op', type=str, default='mean', help='Subword Reconstruction Strategy', required=False, choices=['mean', 'first', 'sum'])
    parser.add_argument('-layers', type=str, default='-1 -2 -3 -4', help='Relevant NLM layers', required=False)
    parser.add_argument('-layer_op', type=str, default='sum', help='Operation to combine layers', required=False,
                        choices=['mean', 'max', 'sum', 'concat', 'ws'])
    parser.add_argument('-max_instances', type=float, default=float('inf'), help='Maximum number of examples for each sense', required=False)
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

    if args.dataset == 'semcor':
        train_path = args.eval_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
        keys_path = args.eval_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
    elif args.dataset == 'semcor_uwa10':
        train_path = 'external/wsd_eval/WSD_Evaluation_Framework/Training_Corpora/SemCor+UWA10/semcor+uwa10.data.xml'
        keys_path = 'external/wsd_eval/WSD_Evaluation_Framework/Training_Corpora/SemCor+UWA10/semcor+uwa10.gold.key.txt'

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


    """
    Pre-processing mapping between sensekeys and synsets.
    """
    map_sk2syn = {}
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            map_sk2syn[lemma.key()] = synset.name()


    gen_vecs(args, encoder, train_path, keys_path)
