# Adapted from:
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Developed with torch.__version__ == 1.5.0

import torch as th
import numpy as np
from typing import List
from collections import Counter


def align_features_to_words(roberta, features, alignment, model_name=None):
    """
    Align given features to words.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = th.stack(output)
    # assert th.all(th.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)

    return output


def align_bpe_to_words(roberta, bpe_tokens: th.LongTensor, other_tokens: List[str], model_name=None):
    """
    Helper to align GPT-2 BPE to other tokenization formats

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`

    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    assert bpe_tokens.dim() == 1
    assert bpe_tokens[0] == 0

    def clean(text):
        return text.strip()

    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]    
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]

    # strip leading <s>
    bpe_tokens = bpe_tokens[1:]

    other_tokens = [clean(str(o)) for o in other_tokens]
    assert ''.join(bpe_tokens) == ''.join(other_tokens)

    # create alignment from every word to a list of BPE tokens
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok):]
                other_tok = ''
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == '':
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)

    return alignment


class FairSeqEncoder():

    def __init__(self, config):
        self.config = config
        self.nlm_model = None
        self.pad_idx = None
        self.nlm_weights = []

        self.load_nlm(self.config['model_name_or_path'])

        if config['weights_path'] != '':
            self.load_layer_weights(config['weights_path'])

    def load_layer_weights(self, weights_path):
        self.nlm_weights = []
        with open(weights_path) as f:
            for line in f:
                self.nlm_weights.append(float(line.strip()))
        self.nlm_weights = th.tensor(self.nlm_weights).to('cuda')
        # self.nlm_weights = np.array(self.nlm_weights)

    def load_nlm(self, model_name):
        model_name = model_name.replace('-', '.')
        self.nlm_model = th.hub.load('pytorch/fairseq', model_name)

        self.nlm_model.eval()
        self.nlm_model.cuda()
        self.pad_idx = self.nlm_model.task.source_dictionary.pad()

    def add_padding_encodings(self, encodings, max_len):
        padding = th.tensor([self.pad_idx] * (max_len - len(encodings)), dtype=th.long)
        return th.cat((encodings, padding))

    def align_toks(self, bpe, toks):
        return align_bpe_to_words(self.nlm_model, bpe, toks, model_name=self.config['model_name_or_path'])

    def align_feats(self, feats, alignment):
        return align_features_to_words(self.nlm_model, feats, alignment, model_name=self.config['model_name_or_path'])

    def get_encodings(self, toks):
        return self.nlm_model.encode(' '.join(toks))

    def is_valid(self, toks):
        encodings = self.get_encodings(toks)
        if len(encodings) > self.config['max_seq_len']:
            return False
        else:
            return True

    def get_num_subtokens(self, toks):
        return len(self.get_encodings(toks))

    def token_embeddings(self, batch_toks, return_tokens=True):

        batch_bpe = [self.get_encodings(toks) for toks in batch_toks] 
        batch_aln = [self.align_toks(bpe, toks) for bpe, toks in zip(batch_bpe, batch_toks)]

        # prepare for model
        input_ids = []
        batch_max_len = max([len(e) for e in batch_bpe])
        for enc in batch_bpe:
            input_ids.append(self.add_padding_encodings(enc, batch_max_len))
        input_ids = th.stack(input_ids)

        # get features
        with th.no_grad():
            batch_hidden_states = self.nlm_model.extract_features(input_ids, return_all_hiddens=True)
        sel_hidden_states = [batch_hidden_states[i] for i in self.config['layers']]

        # combine layers
        if len(sel_hidden_states) > 1:
            sel_hidden_states = th.stack(sel_hidden_states)

            if self.config['layer_op'] == 'sum':
                sel_hidden_states = sel_hidden_states.sum(axis=0)
            elif self.config['layer_op'] == 'mean':
                sel_hidden_states = sel_hidden_states.mean(axis=0)
            elif self.config['layer_op'] == 'max':
                sel_hidden_states = sel_hidden_states.max(axis=0).values
            elif self.config['layer_op'] == 'concat':
                sel_hidden_states = sel_hidden_states.reshape((sel_hidden_states.shape[1], -1)) 
            elif self.config['layer_op'] == 'ws':
                sel_hidden_states = [w*m for w, m in zip(self.nlm_weights, sel_hidden_states)]
                sel_hidden_states = th.stack(sel_hidden_states)
                sel_hidden_states = sel_hidden_states.sum(axis=0)
                # sel_hidden_states = self.nlm_weights.dot(sel_hidden_states)
        else:
            sel_hidden_states = sel_hidden_states[0]

        # align layers
        batch_embeddings = []
        for sent_idx, sent_embeddings in enumerate(sel_hidden_states):
            sent_embeddings = self.align_feats(sent_embeddings, batch_aln[sent_idx])
            sent_embeddings = sent_embeddings[1:-1]  # ignoring special tokens
            sent_tokens = batch_toks[sent_idx]

            paired_embeddings = []
            for tok, emb in zip(sent_tokens, sent_embeddings):
                emb = emb.detach().cpu().numpy()
                paired_embeddings.append((tok, emb))

            batch_embeddings.append(paired_embeddings)
        
        return batch_embeddings


if __name__ == '__main__':

    encoder_cfg = {
        'model_name_or_path': 'roberta-base',
        'weights_path': '',
        'min_seq_len': 0,
        'max_seq_len': 512,
        'layers': [-1, -2, -3, -4],
        'layer_op': 'sum',
        'subword_op': 'mean'
    }

    enc = FairSeqEncoder(encoder_cfg)

    tokenized_sents = [['Hello', 'world', '!'], ['Hello', 'world', ',', 'see', 'you', 'later', '?']]
    
    r = enc.token_embeddings(tokenized_sents)
