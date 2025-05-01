
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import numpy.testing as npt

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

def test_trend_with_xarray():
    # Test the trend function with an xarray DataArray
    data = xr.DataArray(np.array([1, 2, 3, 4, 5]), dims='time')

    # Call the trend function
    slope, intercept, p = trend(data)

    # Check the results
    assert slope == pytest.approx(1.0)
    assert p == pytest.approx(0.0, abs=1e-1)

def test_trend_with_xarray_with_multiple_coords():

    # Let's create some random data
    data = np.arange(1, 11).repeat(25).reshape(10, 5, 5)
    longitude = np.arange(30, 35, 1)
    latitude = np.arange(20, 25, 1)
    time = pd.date_range("2020-01-01", periods=10)

    da = xr.DataArray(
                        data,
                        coords={"lon": longitude, 
                                "lat": latitude, 
                                "time": time},
                        dims=("time", "lat", "lon")
                    )

    result = xr.apply_ufunc(
        trend,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float, float, float]
        )
  
    # Check the results
    expected_slope = np.array([[1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.]])

    expected_intercept = np.array([[1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1.]])

    expected_p = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

    # Slope
    npt.assert_array_equal(result[0].values, expected_slope)

    # Intercept
    npt.assert_array_equal(result[1].values, expected_intercept)

    # P-value
    npt.assert_allclose(result[2].values, expected_p, atol=1e-1)
