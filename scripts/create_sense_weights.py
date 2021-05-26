import sys
import numpy as np

t_str = sys.argv[1]
t = float(t_str)  # temperature parameter

# expects file with rows for each models, with a sequence of scores per layer (should be in repo)
# example below:
# bert-base-cased 52.7 57.8 63.2 66.2 67.2 68.1 69.2 69.7 70.1 70.6 71.7 72.5 71.3
# bert-base-uncased 52.8 60.7 63.9 66.4 67.6 69.1 69.6 70.0 70.5 71.0 72.2 73.0 72.1
# ...
scores_path = 'data/nlm_single_layer_val_f1.txt'

with open(scores_path) as f:
    for line in f:
        elems = line.split()
        nlm_id, scores = elems[0], elems[1:]
        # scores = np.array([float(v) for v in scores], dtype=np.double)
        scores = np.array([float(v) * 0.01 for v in scores], dtype=np.double)

        weights = np.exp(scores/t)
        weights = weights/np.sum(weights)
        weights = ['%.5f' % w for w in weights]

        print('%s %s' % (nlm_id, ' '.join(weights)))
