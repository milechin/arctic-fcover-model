
import pytest
import numpy as np
from T04_lc_trends import trend


def test_trend():
    # Test the trend function
    # Create a small test dataset
    data = np.array([1, 2, 3, 4, 5])

    # Call the trend function
    slope, intercept, p = trend(data)

    # Check the results
    # The slope should be 1.0 for this simple case
    assert slope == pytest.approx(1.0)
    # The p-value should be 0.0 for this simple case
    assert p == pytest.approx(0.0, abs=1e-1)


def test_trend_with_nan():
    # Test the trend function with NaN values
    data = np.array([1, 2, np.nan, 4, 5])

    # Call the trend function
    slope, intercept, p = trend(data)

    # Check the results
    # The slope should be 1.0 for this simple case
    assert slope == pytest.approx(1.0)
    # The p-value should be 0.0 for this simple case
    assert p == pytest.approx(0.0, abs=1e-1)


def test_trend_with_empty_data():
    # Test the trend function with empty data
    data = np.array([])

    # Call the trend function
    slope, intercept, p = trend(data)

    # Check the results
    assert slope == -9999
    assert intercept == -9999
    assert p == -9999


def test_trend_with_all_nan():
    # Test the trend function with all NaN values
    data = np.array([np.nan, np.nan, np.nan])

    # Call the trend function
    slope, intercept, p = trend(data)

    # Check the results
    assert slope == -8888
    assert intercept == -8888
    assert p == -8888
