import inspect
import pytest

import numpy as np
import tabulate
from contexttimer import Timer
from sklearn import neighbors

import pynanoflann


def test(search_type='knn', data_dim=3, n_index_points=2000, n_query_points=100, n_neighbors=10, metric='l2', output=False, radius=1, root_dist=True):
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
        if root_dist:
            nn = pynanoflann.KDTree(n_neighbors=n_neighbors, metric=metric, radius=radius, root_dist=root_dist)
        else:
            nn = pynanoflann.KDTree(n_neighbors=n_neighbors, metric=metric, radius=radius**2, root_dist=root_dist)

        nn.fit(data)

    with Timer() as kd_query:
        if search_type == 'knn':
            kd_res_dist, kd_res_idx = nn.kneighbors(queries)
        else:
            kd_res_dist, kd_res_idx = nn.radius_neighbors(queries)

    # allow small diff due to floating point computation
    params = {}
    for k in inspect.signature(test).parameters:
        params[k] = locals().get(k)

    if not root_dist:
        if search_type == 'knn':
            kd_res_dist = np.sqrt(kd_res_dist)
        else:
            kd_res_dist = [np.sqrt(np.array(x)) for x in kd_res_dist]
        
    if search_type == 'knn':
        assert (kd_res_idx == sk_res_idx).mean() > 0.99, params
        assert np.allclose(kd_res_dist, sk_res_dist), params
    else:
        # sklearn radius search does not allow to return sorted indices
        # So let's check as an unordered sets
        for k, s in zip(kd_res_idx, sk_res_idx):
            if len(k):
                rat = len(set(k).intersection(set(s))) / len(k)
                assert rat > 0.99
            else:
                assert (k == s).all()

    if output and search_type == 'knn':
        diff = kd_res_dist - sk_res_dist
        data = [['sk', sk_init, sk_query], ['kd', kd_init, kd_query]]
        t = tabulate.tabulate(data, headers=['', 'Init', 'Query'], tablefmt='psql')
        print(t)
        print('Dist diff: {}'.format(diff.sum()))
        print('IDX diff: {} / {}'.format((kd_res_idx != sk_res_idx).sum(), kd_res_idx.size))


def test_dimensions():
    for dim in range(1, 10):
        test(data_dim=dim, n_index_points=2000, n_query_points=100)


def test_metric():
    for m in ['l1', 'l2']:
        test(metric=m)


def test_search_type():
    for t in ['knn', 'radius_search']:
        test(search_type=t, radius=100)


def test_incorrect_param():
    with pytest.raises(ValueError):
        nn = pynanoflann.KDTree(metric='l3')

    nn = pynanoflann.KDTree(n_neighbors=10)
    with pytest.raises(ValueError):
        nn.fit(np.array(['str', 'qwe']))
    with pytest.raises(ValueError):
        nn.fit(np.random.uniform(size=(1, 2, 3)))

    with pytest.raises(ValueError):
        nn.fit(np.random.uniform(size=(5, 10)))
        nn.kneighbors(np.random.uniform(size=(2, 10)))


def test_radius():
    nn = pynanoflann.KDTree(metric='l1', radius=1)
    nn.fit(np.array([[1.], [2.], [3.], [4.]]).reshape((-1, 1)))
    distances, indices = nn.radius_neighbors(np.array([[1.5]]).reshape((-1, 1)))
    assert set(indices[0]) == {0, 1}

    distances, indices = nn.radius_neighbors(np.array([[1.5]]).reshape((-1, 1)), radius=0.1)
    assert set(indices[0]) == set()


def test_radius_arg_passing():
    nn = pynanoflann.KDTree(metric='l2', radius=2)
    index = np.array([[1.], [2.], [3.], [4.]]).reshape(-1, 1)
    nn.fit(index)
    query = np.array([[0.1]]).reshape(-1, 1)
    _, indices1 = nn.radius_neighbors(query)
    _, indices2 = nn.radius_neighbors(query, radius=2)
    assert (indices1[0] == indices2[0]).all()
    assert set(indices1[0]) == {0, 1}


def test_warning():
    with pytest.warns(Warning):
        nn = pynanoflann.KDTree()
        nn.fit(np.random.uniform(size=(100, 100)))


def test_consistency_with_sklearn():
    test(data_dim=4, n_index_points=1000000, n_query_points=50000, n_neighbors=10, metric='l2', output=True)
    test(data_dim=4, n_index_points=1000000, n_query_points=50000, n_neighbors=10, metric='l2', output=True, root_dist=False)


if __name__ == '__main__':
    # test(data_dim=4, n_index_points=1000000, n_query_points=50000, n_neighbors=10, metric='l2', output=True)
    test_radius_arg_passing()
