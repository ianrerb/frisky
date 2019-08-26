import numpy as np


def get_whitening_transform(fcov):
    """
    returns R in fcov = R*R^T used to whiten loadings matrices
    """
    u, s, _ = np.linalg.svd(fcov)
    if len(s.shape) == 1:
        return u * s ** .5
    elif len(s.shape) == 2:
        return np.einsum('abc, ac-> abc', u, s ** .5)
    else:
        raise ValueError("invalid number of fcov dimensions.")
