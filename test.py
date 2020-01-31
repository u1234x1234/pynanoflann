import numpy as np

import time
import sys
sys.path.append('./')
import nanoflann
from sklearn import neighbors
from contextlib import contextmanager
from uxils.profiling import Profiler
from contexttimer import Timer
import tabulate


def test(dim=4, n_index_points=1000000, n_query_points=50000, n_neighbors=10, output=False):
    data = np.random.uniform(0, 100, size=(n_index_points, dim)).astype(np.float32)
    queries = np.random.uniform(0, 100, size=(n_query_points, dim)).astype(np.float32)

    with Timer() as sk_init:
        nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nn.fit(data)
    with Timer() as sk_query:
        sk_res_dist, sk_res_idx = nn.kneighbors(queries)


    with Timer() as kd_init:
        index = nanoflann.KDTree(data)
    with Timer() as kd_query:
        kd_res_dist, kd_res_idx = index.query(queries, n_neighbors=n_neighbors)


    diff = kd_res_dist - (sk_res_dist ** 2)

    if output:
        data = [['sk', sk_init, sk_query], ['kd', kd_init, kd_query]]
        t = tabulate.tabulate(data, headers=['', 'Init', 'Query'], tablefmt='psql')
        print(t)
        print('Dist diff: {}'.format((diff ** 2).sum()))
        print('IDX diff: {} / {}'.format((kd_res_idx != sk_res_idx).sum(), kd_res_idx.size))

    # allow small diff due to floating point computation
    params = f'dim={dim}, n_index_points={n_index_points}, '\
        f'n_query_points={n_query_points}, n_neighbors={n_neighbors}'
    assert (kd_res_idx == sk_res_idx).mean() > 0.999, params
    assert np.allclose(kd_res_dist, sk_res_dist**2), params


if __name__ == '__main__':
    for dim in range(1, 10):
        test(dim=dim, n_index_points=2000, n_query_points=100)
    print('oK')
    test(output=True)
