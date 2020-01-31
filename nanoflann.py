import numpy as np
import cppimport
import logging

SUPPORTED_TYPES = [np.float32, np.float64]


def _check_arg(points):
    if points.dtype not in SUPPORTED_TYPES:
        raise ValueError('Supported types: [{}]'.format(SUPPORTED_TYPES))
    if len(points.shape) != 2:
        raise ValueError(f'Incorrect shape {len(points.shape)} != 2')


class KDTree:
    def __init__(self, points):
        _check_arg(points)
        if points.shape[1] > 64:
            logging.warning('KD Tree structure is not a good choice for high dimensional spaces.'
                            'Consider a more suitable search structure.')

        pynanoflann = cppimport.imp('pynanoflann')
        self.index = pynanoflann.KDTree(points)

    def query(self, points, n_neighbors=10):
        _check_arg(points)
        if points.shape[0] < n_neighbors:
            raise ValueError(f"Expected n_neighbors <= n_samples,\
                 but n_samples = {points.shape[0]}, n_neighbors = {n_neighbors}")

        results = self.index.query(points, n_neighbors)
        return results
