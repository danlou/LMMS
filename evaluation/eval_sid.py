"""
Example usage:
python evaluation/eval_sid.py data/vectors/albert-xxlarge-v2/lmms-sp-usm.albert-xxlarge-v2.synsets.300d.vectors.txt

Note: In case of UnicodeDecodeError, run $ export LC_ALL="en_US.UTF-8"
"""

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# from scipy.stats import spearmanr
from scipy.stats import pearsonr
from vectorspace import VSM


def get_correlation(vsm, sid_path):

    n_skipped = 0
    ranked_gold, ranked_pred = [], []
    with open(sid_path) as SID_f:
        for line_idx, line in enumerate(SID_f):
            w1, w2, score, bns1, bns2, wns1, wns2 = line.strip().split('\t')

            wns1 = [wn1 for wn1 in wns1.split(';') if wn1 in vsm.labels_set]
            wns2 = [wn2 for wn2 in wns2.split(';') if wn2 in vsm.labels_set]
            if len(wns1) == 0 or len(wns2) == 0:
                print('L:%d contains OOV - %s' % (line_idx+1, line.strip()))
                n_skipped += 1
                continue
            
            pair_sim = -1
            for wn1 in wns1:
                for wn2 in wns2:
                    pair_sim = max(pair_sim, vsm.similarity(wn1, wn2))
            
            ranked_pred.append((pair_sim, line_idx, w1, w2))
            ranked_gold.append((float(score), line_idx, w1, w2))
            # print(line_idx, wn1, wn2, pred_score)

    ranked_gold.sort(key=lambda x: x[0])
    ranked_pred.sort(key=lambda x: x[0])

    gold_pairs = [(w1, w2) for (score, idx, w1, w2) in ranked_gold]
    pred_pairs = [(w1, w2) for (score, idx, w1, w2) in ranked_pred]
    gold_ranks = list(range(len(gold_pairs)))
    pred_ranks = [gold_pairs.index(pred_pair) for pred_pair in pred_pairs]
    # rho, pval = spearmanr(gold_ranks, pred_ranks)
    corr, pval = pearsonr(gold_ranks, pred_ranks)

    # print('n=%d, r=%f, pval=%f' % (len(gold_ranks), corr, pval))
    if n_skipped > 0:
        print('%d pairs skipped' % n_skipped)

    return corr


if __name__ == '__main__':

    vsm_path = sys.argv[1]
    print(vsm_path)

    vsm = VSM()
    vsm.load_txt(vsm_path)
    vsm.normalize()

    corr = get_correlation(vsm, 'external/SID/SID_subset_wn_full.tsv')
    print('WN Full - %f' % corr)

    corr = get_correlation(vsm, 'external/SID/SID_subset_wn.tsv')
    print('WN Overlapping - %f' % corr)

    corr = get_correlation(vsm, 'external/SID/SID_subset_wn_pol.tsv')
    print('WN Overlapping Polarized - %f' % corr)

    corr = get_correlation(vsm, 'external/SID/SID_subset_wn_obs.tsv')
    print('WN Overlapping Observed - %f' % corr)
