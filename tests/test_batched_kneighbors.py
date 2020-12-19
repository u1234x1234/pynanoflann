

def test_batched():
    import pynanoflann
    import numpy as np

    n_batches = 100
    target = np.random.rand(n_batches, 10000, 3).astype(np.float32)
    query = np.random.rand(n_batches, 2000, 3).astype(np.float32)

    g_res_d = []
    g_res_i = []
    for i in range(n_batches):
        kd_tree = pynanoflann.KDTree(n_neighbors=4, metric='L2', leaf_size=20)
        kd_tree.fit(target[i])
        d, nn_idx = kd_tree.kneighbors(query[i])
        g_res_d.append(d)
        g_res_i.append(nn_idx)

    g_res_d = np.array(g_res_d)
    g_res_i = np.array(g_res_i)

    distances, indices = pynanoflann.batched_kneighbors(target, query, n_neighbors=4, metric='L2', leaf_size=20, n_jobs=2)
    distances = np.array(distances)
    indices = np.array(indices)

    assert np.allclose(g_res_d, distances)
    assert (indices == g_res_i).all()
