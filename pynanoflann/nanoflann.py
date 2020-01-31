"Sklearn interface to the native nanoflann module"
import logging
import numpy as np
from sklearn.neighbors.base import (KNeighborsMixin, NeighborsBase,
                                    RadiusNeighborsMixin, UnsupervisedMixin)
from sklearn.utils.validation import check_is_fitted

import pynanoflann_ext

# TODO add np.float64 support
SUPPORTED_TYPES = [np.float32]


def _check_arg(points):
    if points.dtype not in SUPPORTED_TYPES:
        raise ValueError('Supported types: [{}]'.format(SUPPORTED_TYPES))
    if len(points.shape) != 2:
        raise ValueError(f'Incorrect shape {len(points.shape)} != 2')


class KDTree(NeighborsBase, KNeighborsMixin,
             RadiusNeighborsMixin, UnsupervisedMixin):

    def __init__(self, n_neighbors=5, radius=1.0,
                 leaf_size=10, metric='l2'):
        if metric == 'l2':  # nanoflann uses squared distances
            radius = radius ** 2

        super().__init__(
            n_neighbors=n_neighbors, radius=radius,
            leaf_size=leaf_size, metric=metric)

        if metric not in ['l1', 'l2']:
            raise ValueError('Supported metrics: ["l1", "l2"]')

        self.index = pynanoflann_ext.KDTree(n_neighbors, leaf_size, metric, radius)

    def _fit(self, points: np.ndarray):
        _check_arg(points)
        if points.shape[1] > 64:
            logging.warning('KD Tree structure is not a good choice for high dimensional spaces.'
                            'Consider a more suitable search structure.')
        self.index.fit(points)
        self._fit_X = points

    def kneighbors(self, X, n_neighbors=None):
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if X.shape[0] < n_neighbors:
            raise ValueError(f"Expected n_neighbors <= n_samples,\
                 but n_samples = {points.shape[0]}, n_neighbors = {n_neighbors}")

        dists, idxs = self.index.kneighbors(X, n_neighbors)
        if self.metric == 'l2':  # nanoflann returns squared
            dists = np.sqrt(dists)

        return dists, idxs

    def radius_neighbors(self, X, radius=None, return_distance=True):
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if radius is None:
            radius = self.radius

        dists, idxs = self.index.radius_neighbors(X, radius)
        idxs = np.array([np.array(x) for x in idxs])

        if self.metric == 'l2':  # nanoflann returns squared
            dists = np.array([np.sqrt(np.array(x)).squeeze() for x in dists])
        else:
            dists = np.array([np.array(x).squeeze() for x in dists])

        if return_distance:
            return dists, idxs
        else:
            return idxs
