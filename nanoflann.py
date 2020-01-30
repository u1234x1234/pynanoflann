import numpy as np
import cppimport


SUPPORTED_TYPES = [np.float32, np.float64]


def _check_arg(points):
    if points.dtype not in SUPPORTED_TYPES:
        raise ValueError('Supported types: [{}]'.format(SUPPORTED_TYPES))
    if len(points.shape) != 2 or points.shape[1] not in [2, 3]:
        raise ValueError('Incorrect shape {}. Supported shapes: [x, 2] or [x, 3]')


class KDTree:
    def __init__(self, points):
        _check_arg(points)

        pynanoflann = cppimport.imp('pynanoflann')
        self.index = pynanoflann.KDTree(points)

    def query(self, points, n_neighbors=10):
        _check_arg(points)

        # points = np.ascontiguousarray(points).astype(np.float32)

        r = self.index.query(points, n_neighbors)
        return r
