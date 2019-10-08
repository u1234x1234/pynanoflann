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


with timing('sklearn init'):
    nn = neighbors.NearestNeighbors(n_neighbors=200)
    nn.fit(data)

with timing('sklearn query'):
    sk_result = nn.kneighbors([[0.5, 0.5, 0.5]])


with timing('kd init'):
    index = nanoflann.KDTree(data)

with timing('kd query'):
    kd_result = index.query(data, n_neighbors=200)


assert (kd_result[1] == sk_result[1]).all() # idxs
assert np.allclose(kd_result[0], (sk_result[0] ** 2))
