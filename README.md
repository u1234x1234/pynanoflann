[![Build Status](https://travis-ci.org/u1234x1234/pynanoflann.svg?branch=master)](https://travis-ci.org/u1234x1234/pynanoflann)
[![codecov](https://codecov.io/gh/u1234x1234/pynanoflann/branch/master/graph/badge.svg)](https://codecov.io/gh/u1234x1234/pynanoflann)

# pynanoflann

Unofficial python wrapper to the [nanoflann](https://github.com/jlblancoc/nanoflann) library [1] with sklearn interface and additional multithreaded capabilities.

nanoflann implementation of [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) provides one of the best performance for many k-nearest neighbour problems [2].

It is a good choice for exact k-NN, radius searches in low dimensional spaces.

# Install

```
pip install git+https://github.com/u1234x1234/pynanoflann.git@0.0.7
```

# Basic Usage

```python
import numpy as np
import pynanoflann

index = np.random.uniform(0, 100, (100, 4))
queries = np.random.uniform(0, 100, (10, 4))

nn = pynanoflann.KDTree(n_neighbors=5, metric='L1', radius=100)
nn.fit(index)

# Get k-nearest neighbors
distances, indices = nn.kneighbors(queries)

# Radius search
distances, indices = nn.radius_neighbors(queries)

```

# Save, load

If you need to save the model, there are two options:

1. Save only the built index. In this case, data points are NOT stored in file. Very efficient, but inconvenient in some cases.
```python

kdtree.fit(X)
kdtree.save_index('index.bin')

prebuilt_kdtree = pynanoflann.KDTree()
# Must use the same data on which the index was built.
prebuilt_kdtree.fit(X, 'index.bin')
```

Please refer to the detailed [example](https://github.com/u1234x1234/pynanoflann/blob/master/tests/test_save_load.py#L8)

2. Pickle the whole model with data points. Less efficient, but convenient.
```python
kdtree.fit(X)
with open('kdtree.pkl', 'wb') as out_file:
    pickle.dump(kdtree, out_file)
with open('kdtree.pkl', 'rb') as in_file:
    unpickled_kdtree = pickle.load(in_file)
```
Please refer to the detailed [example](https://github.com/u1234x1234/pynanoflann/blob/master/tests/test_save_load.py#L43)

# Multicore parallelization

* Query parallelization:
[Example](https://github.com/u1234x1234/pynanoflann/blob/master/tests/test_multithreaded.py)

* Simultaneous indexing+querying parallelization:
[Example](https://github.com/u1234x1234/pynanoflann/blob/master/tests/test_batched_kneighbors.py),
[Discussion](https://github.com/u1234x1234/pynanoflann/issues/3)

# Performance

Generally it much faster than brute force or [cython implementation](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/_kd_tree.pyx) of k-d tree in sklearn

To run benchmark:
```
python benchmark.py
```

<img src="https://i.imgur.com/6Y6VrZb.png" width="1200">

<img src="https://i.imgur.com/c7OGvV8.png" width="1200">


# Links

1. Blanco, Jose Luis and Rai, Pranjal Kumar, 2014, nanoflann: a C++ header-only fork of FLANN, a library for Nearest Neighbor ({NN}) with KD-trees.
2. Vermeulen, J.L., Hillebrand, A. and Geraerts, R., 2017. A comparative study of k‐nearest neighbour techniques in crowd simulation.
