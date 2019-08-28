import numpy as np


def get_whitening_transform(fcov):
    """
    returns R in fcov = R*R^T used to whiten loadings matrices
    """
    u, s, _ = np.linalg.svd(fcov)
    dim = len(s.shape)
    if dim == 1:
        return u * s ** 0.5
    elif dim == 2:
        return np.einsum("abc, ac-> abc", u, s ** 0.5)
    else:
        raise ValueError("invalid number of fcov dimensions.")


def risk_exposures(weights, rdata):
    mult = weights * rdata
    mult["common"] = mult.common.sum("security")
    return mult


def sigma_w(weights, rdata):
    result = (rdata * risk_exposures(weights, rdata)).sum("factor")
    return result.common + result.specific


def covariance(weights, other, rdata):
    return (other * sigma_w(weights, rdata)).sum("security")


def variance(weights, rdata):
    return covariance(weights, weights, rdata)


def volatility(weights, rdata):
    return variance(weights, rdata) ** 0.5


def portfolio_beta(weights, index_weights, rdata):
    sw = sigma_w(index_weights, rdata)
    var = (index_weights * sw).sum("security")
    cov = (weights * sw).sum("security")
    return cov / var


def risk_contribution(weights, rdata):
    sw = sigma_w(weights, rdata)
    return sw / (weights * sw).sum("security") ** 0.5
