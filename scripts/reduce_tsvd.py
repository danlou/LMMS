import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from vectorspace import VSM

vecs_path = sys.argv[1]

rank = 300
vecs_svd_path = vecs_path.replace('.txt', '.svd_%d.txt' % rank)

print(datetime.now(), 'Loading %s ...' % vecs_path)
vsm = VSM()
vsm.load_txt(vecs_path)

print(datetime.now(), 'Applying TSVD (rank=%d) ...' % rank)
tsvd = TruncatedSVD(n_components=rank, random_state=42)
vsm.vectors = tsvd.fit_transform(vsm.vectors)

print(datetime.now(), 'Writing %s ...' % vecs_svd_path)
with open(vecs_svd_path, 'w') as svd_f:
    for label, vec in zip(vsm.labels, vsm.vectors):
        vec_str = ' '.join([str(round(v, 6)) for v in vec.tolist()])
        svd_f.write('%s %s\n' % (label, vec_str))
