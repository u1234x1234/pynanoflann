import numpy as np

import time
import sys
sys.path.append('./')
import nanoflann
from sklearn import neighbors
from contextlib import contextmanager
from uxils.profiling import Profiler


@contextmanager
def timing(description):
    start = time.time()
    yield
    ellapsed_time = time.time() - start

    print(f'{description}: {ellapsed_time}')


def test(dim=3, n_neighbors=10):
    data = np.random.uniform(0, 100, size=(10000000, dim)).astype(np.float32)
    queries = np.random.uniform(0, 100, size=(50000, dim)).astype(np.float32)

    with timing('sklearn init'):
        nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nn.fit(data)

    with timing('sklearn query'):
        sk_res_dist, sk_res_idx = nn.kneighbors(queries)


    with timing('kd init'):
        index = nanoflann.KDTree(data)

    with timing('kd query'):
        kd_res_dist, kd_res_idx = index.query(queries, n_neighbors=n_neighbors)


    print('IDX diff: {} / {}'.format((kd_res_idx != sk_res_idx).sum(), kd_res_idx.size))
    diff = kd_res_dist - (sk_res_dist ** 2)
    print('Dist diff: {}'.format((diff ** 2).sum()))

    # allow small diff due to floating point computation
    assert (kd_res_idx == sk_res_idx).mean() > 0.999
    assert np.allclose(kd_res_dist, sk_res_dist**2)


if __name__ == '__main__':
    test()
