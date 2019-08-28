import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal

from frisky.core import calc

NUM_FACTORS = 2
NP_SEED = 42
NUM_SECS = 5
NUM_DATES = 10


@pytest.fixture(scope="module")
def single_fcov_array():
    return np.array([[2.0, -3.0], [-3.0, 6.0]])


@pytest.fixture(scope="module")
def sec_index():
    return pd.Index([f"s{x}" for x in range(NUM_SECS)], name="security")


@pytest.fixture(scope="module")
def date_index():
    return pd.date_range("2014-04-17", periods=NUM_DATES, freq="B", name="date")


@pytest.fixture(scope="module")
def factor_index():
    return pd.Index(range(NUM_FACTORS), name="factor")


@pytest.fixture(scope="module")
def rdata(sec_index, date_index, factor_index):
    ld_array = xr.DataArray(
        np.random.randn(NUM_DATES, NUM_SECS, NUM_FACTORS),
        coords=[date_index, sec_index, factor_index],
        dims=["date", "security", "factor"],
    )

    spec_array = xr.DataArray(
        np.random.uniform(size=(NUM_DATES, NUM_SECS)),
        coords=[date_index, sec_index],
        dims=["date", "security"],
    )

    return xr.Dataset({"common": ld_array, "specific": spec_array})


@pytest.fixture(scope="module")
def equal_weights(sec_index, date_index):
    return pd.DataFrame(1.0 / NUM_SECS, date_index, sec_index).stack().to_xarray()


def test_whitening_transform(single_fcov_array):
    """
    test that fcov = R*R^T
    """
    transform = calc.get_whitening_transform(single_fcov_array)
    assert_almost_equal(transform.dot(transform.T), single_fcov_array)


def test_whitening_transform_batch(single_fcov_array):
    """
    test transform on repeated sequence of 
    factor covariance matrices produces repeated transforms
    """
    n_repeats = 3

    trans = calc.get_whitening_transform(single_fcov_array)
    batched = calc.get_whitening_transform(
        np.array([single_fcov_array for x in range(n_repeats)])
    )

    assert_almost_equal(np.array([trans for x in range(n_repeats)]), batched)


def test_risk_exposures_single_security(rdata, sec_index, date_index):
    for sec in sec_index:
        weights = (
            pd.DataFrame({sec: 1.0}, date_index)
            .reindex(columns=sec_index)
            .stack()
            .to_xarray()
        )

        expo = calc.risk_exposures(weights, rdata)
        #  factor exposure should be loadings for security
        assert_almost_equal(expo.common.values, rdata.common.sel(security=sec).values)

        #  only specific for security
        assert_almost_equal(
            expo.specific.values, rdata.specific.where(rdata.security == sec).values
        )


def test_risk_exposures_equal_weight(rdata, equal_weights):
    expo = calc.risk_exposures(equal_weights, rdata)

    assert_almost_equal(expo.common.values, rdata.common.mean("security").values)

    assert_almost_equal(expo.specific.values, (rdata.specific / NUM_SECS).values)


def test_equal_weight_variance(rdata, equal_weights):
    result = calc.variance(equal_weights, rdata)
    fac_var = (rdata.common.mean("security") ** 2).sum("factor")
    spec_var = ((rdata.specific / NUM_SECS) ** 2).sum("security")
    expected = fac_var + spec_var

    assert_almost_equal(result.values, expected.values)


def test_vol_root_variance(rdata, equal_weights):
    var = calc.variance(equal_weights, rdata)
    vol = calc.volatility(equal_weights, rdata)

    assert_almost_equal(var.values, vol.values ** 2)


def test_covariance_is_variance(rdata, equal_weights):
    var = calc.variance(equal_weights, rdata)
    covar = calc.covariance(equal_weights, equal_weights, rdata)
    assert_almost_equal(var.values, covar.values)


def test_variance_covar_signs(rdata, equal_weights):
    neg_weights = -1.0 * equal_weights
    var = calc.variance(equal_weights, rdata)
    negvar = calc.variance(neg_weights, rdata)
    assert_almost_equal(var.values, negvar.values)

    negcov_1 = calc.covariance(neg_weights, equal_weights, rdata)
    negcov_2 = calc.covariance(equal_weights, neg_weights, rdata)
    assert_almost_equal(negcov_1.values, negcov_2.values)
    assert_almost_equal(negcov_1.values, -1.0 * var.values)


def test_beta_one(rdata, equal_weights):
    result = calc.portfolio_beta(equal_weights, equal_weights, rdata)
    assert (np.abs(result - 1.0) < 1e-6).values.all()


def test_beta_scales(rdata, equal_weights):
    result = calc.portfolio_beta(equal_weights, 2.0 * equal_weights, rdata)
    assert (np.abs(result - 0.5) < 1e-6).values.all()

    result = calc.portfolio_beta(2.0 * equal_weights, equal_weights, rdata)
    assert (np.abs(result - 2.0) < 1e-6).values.all()

    result = calc.portfolio_beta(equal_weights, -1.0 * equal_weights, rdata)
    assert (np.abs(result + 1.0) < 1e-6).values.all()


def test_risk_contrib_vol_equation(rdata, equal_weights):
    contrib = calc.risk_contribution(equal_weights, rdata)
    vol = calc.volatility(equal_weights, rdata)
    result = (equal_weights * contrib).sum("security")

    assert_almost_equal(vol.values, result.values)
