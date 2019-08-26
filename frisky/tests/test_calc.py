import pytest
import numpy as np
from frisky.core import calc

NUM_FACTORS = 2


@pytest.fixture(scope="module")
def single_fcov_array():
    return np.array([[2.0, -3.0], [-3.0, 6.0]])


def test_whitening_transform(single_fcov_array):
    """
    test that fcov = R*R^T
    """
    transform = calc.get_whitening_transform(single_fcov_array)
    np.testing.assert_almost_equal(transform.dot(transform.T), single_fcov_array)


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

    np.testing.assert_almost_equal(np.array([trans for x in range(n_repeats)]), batched)
