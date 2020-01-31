import numpy as np

import inspect
import time
import nanoflann
from sklearn import neighbors

from contexttimer import Timer
import tabulate


def test(search_type='knn', data_dim=3, n_index_points=2000, n_query_points=100, n_neighbors=10, metric='l2', output=False, radius=1):
    data = np.random.uniform(0, 100, size=(n_index_points, data_dim)).astype(np.float32)
    queries = np.random.uniform(0, 100, size=(n_query_points, data_dim)).astype(np.float32)

    with Timer() as sk_init:
        nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric=metric, radius=radius)
        nn.fit(data)
    with Timer() as sk_query:
        if search_type == 'knn':
            sk_res_dist, sk_res_idx = nn.kneighbors(queries)
        else:
            sk_res_dist, sk_res_idx = nn.radius_neighbors(queries)

    with Timer() as kd_init:
        nn = nanoflann.KDTree(n_neighbors=n_neighbors, metric=metric, radius=radius)
        nn.fit(data)
    with Timer() as kd_query:
        if search_type == 'knn':
            kd_res_dist, kd_res_idx = nn.kneighbors(queries)
        else:
            kd_res_dist, kd_res_idx = nn.radius_neighbors(queries)

    if output and search_type == 'knn':
        diff = kd_res_dist - sk_res_dist
        data = [['sk', sk_init, sk_query], ['kd', kd_init, kd_query]]
        t = tabulate.tabulate(data, headers=['', 'Init', 'Query'], tablefmt='psql')
        print(t)
        print('Dist diff: {}'.format(diff.sum()))
        print('IDX diff: {} / {}'.format((kd_res_idx != sk_res_idx).sum(), kd_res_idx.size))

    # allow small diff due to floating point computation
    params = {}
    for k in inspect.signature(test).parameters:
        params[k] = locals().get(k)

    if search_type == 'knn':
        assert (kd_res_idx == sk_res_idx).mean() > 0.999, params
        assert np.allclose(kd_res_dist, sk_res_dist), params
    else:
        # sklearn radius search does not allow to return sorted indices
        # So let's check as an unordered sets
        for k, s in zip(kd_res_idx, sk_res_idx):
            if len(k):
                rat = len(set(k).intersection(set(s))) / len(k)
                assert rat > 0.999
            else:
                assert (k == s).all()


if __name__ == '__main__':
    for dim in range(1, 10):
        test(data_dim=dim, n_index_points=2000, n_query_points=100)
    for m in ['l1', 'l2']:
        test(metric=m)
    for t in ['knn', 'radius_search']:
        test(search_type=t, radius=100)
    test(search_type='radius_search', radius=1)

    print('oK')
    test(data_dim=4, n_index_points=1000000, n_query_points=50000, n_neighbors=10, metric='l2', output=True)
