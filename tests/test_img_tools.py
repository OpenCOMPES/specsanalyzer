"""This is a code that performs several tests for the image tool module"""
import os

import numpy as np
import xarray as xr

from specsanalyzer.config import parse_config
from specsanalyzer.img_tools import crop_xarray
from specsanalyzer.img_tools import fourier_filter_2d

test_dir = os.path.dirname(__file__)

bins2d = (95, 34)
array2d = np.random.normal(size=bins2d)
# strip negative values
for i in range(0, array2d.shape[0]):
    for j in range(0, array2d.shape[1]):
        array2d[i, j] = array2d[i][j] if array2d[i][j] > 0 else 0

angle_axis = np.linspace(0, 1, array2d.shape[0])
ek_axis = np.linspace(0, 1, array2d.shape[1])
da = xr.DataArray(
    data=array2d,
    coords={"Angle": angle_axis, "Ekin": ek_axis},
    dims=["Angle", "Ekin"],
)


def test_fourier_filter_2d():
    """Test if the Fourier filter function returns the same array if no peaks
    are filtered out.
    """
    np.testing.assert_allclose(
        array2d,
        fourier_filter_2d(array2d, []),
        atol=1e-10,
    )

    config_path = os.fspath(f"{test_dir}/data/config/config.yaml")
    data_path = os.fspath(f"{test_dir}/data/dataFHI/Scan1232.tsv")
    filtered_path = os.fspath(f"{test_dir}/data/dataFHI/Scan1232_filtered.tsv")
    config = parse_config(config_path)
    peaks = config["fft_filter_peaks"]
    with open(
        data_path,
        encoding="utf-8",
    ) as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    filtered = fourier_filter_2d(tsv_data, peaks)

    with open(
        filtered_path,
        encoding="utf-8",
    ) as file:
        ref = np.loadtxt(file, delimiter="\t")
    ref = ref.T

    np.testing.assert_allclose(ref, filtered, atol=3.5)


def test_fourier_filter_2d_raises():
    """Test if the Fourier filter function raises an error if a key is not defined."""
    with np.testing.assert_raises(KeyError):
        fourier_filter_2d(array2d, [{"amplitude": 1}])


def test_crop_xarray():
    """Test if the cropping function leaves the xarray intact if we don't crop it."""
    np.testing.assert_allclose(da, crop_xarray(da, 0, 1, 0, 1))
