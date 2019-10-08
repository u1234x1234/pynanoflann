import numpy as np

import time
import sys
sys.path.append('./')
import nanoflann
from sklearn import neighbors
from contextlib import contextmanager


@contextmanager
def timing(description):
    start = time.time()
    yield
    ellapsed_time = time.time() - start

    print(f'{description}: {ellapsed_time}')


data = np.random.uniform(-10, 10, size=(1000000, 3)).astype(np.float32)
queries = np.random.uniform(-10, 10, size=(300, 3)).astype(np.float32)

n_neighbors = 4


with timing('sklearn init'):
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data)

with timing('sklearn query'):
    sk_result = nn.kneighbors(queries)


with timing('kd init'):
    index = nanoflann.KDTree(data)

with timing('kd query'):
    kd_result = index.query(queries, n_neighbors=n_neighbors)


print(sk_result[0].shape, kd_result[0].shape)
print(sk_result[1].shape, kd_result[1].shape)

print(kd_result[1])
print(sk_result[1])

assert (kd_result[1] == sk_result[1]).all() # idxs
assert np.allclose(kd_result[0], (sk_result[0] ** 2))
