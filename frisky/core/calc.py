import numpy as np


def get_whitening_transform(fcov):
    """
    returns R in fcov = R*R^T used to whiten loadings matrices
    """
    u, s, _ = np.linalg.svd(fcov)
    dim = len(s.shape)
    if dim == 1:
        return u * s ** .5
    elif dim == 2:
        return np.einsum('abc, ac-> abc', u, s ** .5)
    else:
        raise ValueError("invalid number of fcov dimensions.")


def risk_exposures(weights, rdata):
    mult = weights * rdata
    mult["common"] = mult.common.sum('security')
    return mult


def covariance(weights, other, rdata):
    result = (
        risk_exposures(weights, rdata) * risk_exposures(other, rdata)
    ).sum("security")
    return result.common + result.specific


def volatility(weights, rdata):
    result = (risk_exposures(weights, rdata) ** 2).sum("security")
    return result.common + result.specific


def sigma_w(weights, rdata):
    result = (risk_exposures(weights, rdata) * rdata).sum('factor')
    return result.common + result.specific
