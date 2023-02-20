"Sklearn interface to the native nanoflann module"
import copyreg
import warnings
from typing import Optional

import nanoflann_ext
import numpy as np
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from sklearn.utils.validation import check_is_fitted

SUPPORTED_TYPES = [np.float32, np.float64]


def pickler(c):
    X = c._fit_X if hasattr(c, "_fit_X") else None
    return unpickler, (c.n_neighbors, c.radius, c.leaf_size, c.metric, X)


def unpickler(n_neighbors, radius, leaf_size, metric, X):
    # Recreate an kd-tree instance
    tree = KDTree(n_neighbors, radius, leaf_size, metric)
    # Unpickling of the fitted instance
    if X is not None:
        tree.fit(X)
    return tree


def _check_arg(points):
    if points.dtype not in SUPPORTED_TYPES:
        raise ValueError("Supported types: [{}]".format(SUPPORTED_TYPES))
    if len(points.shape) != 2:
        raise ValueError(f"Incorrect shape {len(points.shape)} != 2")


class KDTree(NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin):
    def __init__(self, n_neighbors=5, radius=1.0, leaf_size=10, metric="l2", root_dist=True):

        metric = metric.lower()
        if metric not in ["l1", "l2"]:
            raise ValueError('Supported metrics: ["l1", "l2"]')

        if metric == "l2" and root_dist:  # nanoflann uses squared distances
            radius = radius ** 2

        super().__init__(
            n_neighbors=n_neighbors, radius=radius, leaf_size=leaf_size, metric=metric
        )
        self.root_dist = root_dist

    def fit(self, X: np.ndarray, index_path: Optional[str] = None):
        """
        Args:
            X: np.ndarray data to use
            index_path: str Path to a previously built index. Allows you to not rebuild index.
                NOTE: Must use the same data on which the index was built.
        """
        _check_arg(X)
        if X.dtype == np.float32:
            self.index = nanoflann_ext.KDTree32(
                self.n_neighbors, self.leaf_size, self.metric, self.radius
            )
        else:
            self.index = nanoflann_ext.KDTree64(
                self.n_neighbors, self.leaf_size, self.metric, self.radius
            )

        if X.shape[1] > 64:
            warnings.warn(
                "KD Tree structure is not a good choice for high dimensional spaces."
                "Consider a more suitable search structure."
            )

        self.index.fit(X, index_path if index_path is not None else "")
        self._fit_X = X

    def kneighbors(self, X, n_neighbors=None, n_jobs=1):
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if self._fit_X.shape[0] < n_neighbors:
            raise ValueError(
                f"Expected n_neighbors <= n_samples,\
                 but n_samples = {self._fit_X.shape[0]}, n_neighbors = {n_neighbors}"
            )

        if n_jobs == 1:
            dists, idxs = self.index.kneighbors(X, n_neighbors)
        else:
            dists, idxs = self.index.kneighbors_multithreaded(X, n_neighbors, n_jobs)

        if self.metric == "l2" and self.root_dist:  # nanoflann returns squared
            dists = np.sqrt(dists)

        return dists, idxs

    def radius_neighbors(self, X, radius=None, return_distance=True, n_jobs=1):
        check_is_fitted(self, ["_fit_X"], all_or_any=any)
        _check_arg(X)

        if radius is None:
            radius = self.radius
        elif self.metric == "l2" and self.root_dist:
            radius = radius ** 2  # nanoflann internally uses squared distances

        if n_jobs == 1:
            dists, idxs = self.index.radius_neighbors(X, radius)
        else:
            dists, idxs = self.index.radius_neighbors_multithreaded(X, radius, n_jobs)

        idxs = [np.array(x, dtype=np.int64) for x in idxs]

        if self.metric == "l2" and self.root_dist:  # nanoflann returns squared
            dists = [np.sqrt(np.array(x)) for x in dists]
        else:
            dists = [np.array(x) for x in dists]

        if return_distance:
            return dists, idxs

        return idxs

    def get_data(self, copy: bool = True) -> np.ndarray:
        """Returns underlying data points. If copy is `False` then no modifications should be applied to the returned data.

        Args:
            copy: whether to make a copy.
        """
        check_is_fitted(self, ["_fit_X"], all_or_any=any)

        if copy:
            return self._fit_X.copy()
        else:
            return self._fit_X

    def save_index(self, path: str) -> int:
        "Save index to the binary file. NOTE: Data points are NOT stored."
        return self.index.save_index(path)


def batched_kneighbors(
    X_index, X_query, n_neighbors=5, metric="l2", leaf_size=10, n_jobs=1, root_dist=True
):
    """

    Args:
        X_index: list of np.ndarray
        X_query: list of np.ndarray
    """
    metric = metric.lower()
    if metric not in ["l1", "l2"]:
        raise ValueError('Supported metrics: ["l1", "l2"]')
    for x in X_index:
        _check_arg(x)
    for x in X_query:
        _check_arg(x)
    dtypes = list(set([x.dtype for x in X_index] + [x.dtype for x in X_query]))
    assert len(dtypes) == 1, "All data must be the same type"

    if dtypes[0] == np.float32:
        g_d, g_i = nanoflann_ext.batched_kneighbors32(
            list(X_index), list(X_query), n_neighbors, metric, leaf_size, n_jobs
        )
    else:
        g_d, g_i = nanoflann_ext.batched_kneighbors64(
            list(X_index), list(X_query), n_neighbors, metric, leaf_size, n_jobs
        )

    if metric == "l2" and root_dist:  # nanoflann returns squared
        g_d = [np.sqrt(x) for x in g_d]

    return g_d, g_i


# Register pickling of non-trivial types
copyreg.pickle(KDTree, pickler, unpickler)
