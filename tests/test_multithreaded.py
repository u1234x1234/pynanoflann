import pynanoflann
import numpy as np


def test_multithreaded():

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
