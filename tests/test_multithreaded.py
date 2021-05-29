import time

import pynanoflann
import numpy as np


def test_multithreaded_kneighbors():

    n_batches = 10
    target = np.random.rand(n_batches, 10000, 3)
    query = np.random.rand(n_batches, 20000, 3)

    def search_batch(i):
        pts_target = target[i]
        pts_query = query[i]

        kd_tree = pynanoflann.KDTree(n_neighbors=1, metric='L2', leaf_size=20)
        kd_tree.fit(pts_target)

        d, nn_idx = kd_tree.kneighbors(pts_query)
        d2, nn_idx2 = kd_tree.kneighbors(pts_query, n_jobs=4)

        assert np.allclose(d, d2)
        assert (nn_idx == nn_idx2).all()

    list(map(search_batch, range(10)))


def test_multithreaded_radius():
    index = np.random.rand(40_000, 3)
    query = np.random.rand(20_000, 3)

    kd_tree = pynanoflann.KDTree(metric="L2", radius=0.1)
    kd_tree.fit(index)

    t1 = time.time()
    distances1, indices1 = kd_tree.radius_neighbors(query)
    t1 = time.time() - t1

    t2 = time.time()
    distances2, indices2 = kd_tree.radius_neighbors(query, n_jobs=4)
    t2 = time.time() - t2

    assert len(distances1) == len(distances2)
    for d1, d2 in zip(distances1, distances2):
        assert np.allclose(d1, d2)

    assert len(indices1) == len(indices2)
    for i1, i2 in zip(indices1, indices2):
        assert (i1 == i2).all()


if  __name__ == "__main__":
    test_multithreaded_radius()
