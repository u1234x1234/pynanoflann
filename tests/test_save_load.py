import pynanoflann
import numpy as np
import pickle
import os
from contexttimer import Timer


def test_index_save_load():
    data = np.random.uniform(0, 100, size=(500000, 3)).astype(np.float32)
    queries = np.random.uniform(0, 100, size=(100, 3)).astype(np.float32)

    # Lets create an index of kd-tree
    kdtree = pynanoflann.KDTree()
    with Timer() as index_build_time:
        kdtree.fit(data)
    dist1, idx1 = kdtree.kneighbors(queries)

    # Save the built index
    # NOTE: Only the index will be saved, data points are NOT stored in the index
    index_path = '/tmp/index.bin'
    try:
        os.remove(index_path)
    except OSError:
        pass
    kdtree.save_index(index_path)
    assert os.path.exists(index_path)

    # Now, load a prebuilt index
    # BEWARE, data points must be the same
    new_kdtree = pynanoflann.KDTree()
    with Timer() as index_load_time:
        new_kdtree.fit(data, index_path)

    # Fitting with a prebuilt index is much faster, since it only requires loading a binary file 
    assert index_build_time.elapsed > 10 * index_load_time.elapsed

    # At the same time, the results are identical
    dist2, idx2 = kdtree.kneighbors(queries)
    assert (dist2 == dist1).all()
    assert (idx1 == idx2).all()


def test_pickle():
    data = np.random.uniform(0, 100, size=(500000, 3)).astype(np.float32)
    queries = np.random.uniform(0, 100, size=(100, 3)).astype(np.float32)

    leaf_size = 20
    radius = 0.5

    # Construct a kd-tree
    kdtree = pynanoflann.KDTree(metric='l1', leaf_size=leaf_size, radius=radius)
    kdtree.fit(data)
    dist1, idx1 = kdtree.kneighbors(queries)

    # Pickle to memory
    pickled = pickle.dumps(kdtree)

    # Size of the pickled kd-tree includes data points: (500000 points * 3 dim * 4 bytes) ~ 6Mb
    assert 6_000_000 < len(pickled) < 6_001_000

    # Free memory
    del kdtree, data

    # Load a pickled instance
    unpickled_kdtree = pickle.loads(pickled)
    dist2, idx2 = unpickled_kdtree.kneighbors(queries)

    # The results are identical
    assert (dist1 == dist2).all()
    assert (idx1 == idx2).all()

    # Parameters are unpickled correctly
    assert unpickled_kdtree.leaf_size == leaf_size
    assert unpickled_kdtree.radius == radius

    unfitted_kdtree = pynanoflann.KDTree(metric='l1')
    data = pickle.dumps(unfitted_kdtree)
    # Size of the unfitted kd-tree very small: only parameters
    assert len(data) < 200
    un_un_tree = pickle.loads(data)
    assert un_un_tree.metric == 'l1'


test_index_save_load()
test_pickle()
