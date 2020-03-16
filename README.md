[![Build Status](https://travis-ci.org/u1234x1234/pynanoflann.svg?branch=master)](https://travis-ci.org/u1234x1234/pynanoflann)
[![codecov](https://codecov.io/gh/u1234x1234/pynanoflann/branch/master/graph/badge.svg)](https://codecov.io/gh/u1234x1234/pynanoflann)

# pynanoflann

Unofficial python wrapper to the [nanoflann](https://github.com/jlblancoc/nanoflann) library [1] with sklearn interface.

nanoflann implementation of [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) provides one of the best performance for many k-nearest neighbour problems [2].

It is a good choice for exact k-NN, radius searches in low dimensional spaces.

# Install

```
pip install git+https://github.com/u1234x1234/pynanoflann
```

# Usage

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


# Performance

Generally it much faster than brute force or [cython implementation](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/_kd_tree.pyx) of k-d tree in sklearn

To run benchmark:
```
python benchmark.py
```

<img src="https://i.imgur.com/6Y6VrZb.png" width="1200">

<img src="https://i.imgur.com/hbdDIFn.png" width="1200">


# Links

1. Blanco, Jose Luis and Rai, Pranjal Kumar, 2014, nanoflann: a C++ header-only fork of FLANN, a library for Nearest Neighbor ({NN}) with KD-trees.
2. Vermeulen, J.L., Hillebrand, A. and Geraerts, R., 2017. A comparative study of k‐nearest neighbour techniques in crowd simulation.
