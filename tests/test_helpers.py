"""This script performs tests for the helper functions
in the core.py script to support the load_scan method.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import specsscan
from specsscan.core import load_images
from specsscan.core import parse_lut_to_df

package_dir = os.path.dirname(specsscan.__file__)
# Path to a sample 3-D scan
scan_path_mirror = Path(f"{package_dir}/../tests/data/4450")
# Path to a sample 2-D scan
scan_path_single = Path(f"{package_dir}/../tests/data/3610")

df_lut = parse_lut_to_df(scan_path_mirror)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_averaging_with_iterations(delay):
    """Tests the averaging by comparing an AVG scan with
    a corresponding average over all iterations.
    """
    iterations = np.arange(2)
    data_loaded = load_images(scan_path_mirror, df_lut, iterations)
    with open(
        scan_path_mirror.joinpath(f"AVG/00{delay}.tsv"),
        encoding="utf-8",
    ) as file:
        data = np.loadtxt(file, delimiter="\t")
    np.testing.assert_allclose(data_loaded[delay], data)


def test_averaging_with_delays():
    """Tests the check_scan functionality of the
    load_images function by comparing an AVG file
    and the average over all iterations."""

    delay = 0
    data = load_images(scan_path_mirror, df_lut, delays=delay)
    with open(
        scan_path_mirror.joinpath("AVG/000.tsv"),
        encoding="utf-8",
    ) as file:
        data_0 = np.loadtxt(file, delimiter="\t")

    data_avg = np.average(data, axis=0)
    np.testing.assert_allclose(data_avg, data_0)


@pytest.mark.parametrize("scan", [0, 1, 2])
def test_load_averages(scan):
    """Tests loading of 3-D array with default paramteres"""
    data = load_images(scan_path_mirror)
    with open(
        scan_path_mirror.joinpath(f"AVG/00{scan}.tsv"),
        encoding="utf-8",
    ) as file:
        data_avg = np.loadtxt(file, delimiter="\t")
    assert np.array_equal(data[scan], data_avg)


def test_single():
    """Tests loading of single scans with and without iterations"""
    data = load_images(scan_path_single)
    with open(
        scan_path_single.joinpath("AVG/000.tsv"),
        encoding="utf-8",
    ) as file:
        data_0 = np.loadtxt(file, delimiter="\t")
    assert np.array_equal(data[0], data_0)

    with pytest.raises(IndexError):
        load_images(scan_path_single, iterations=[0])


def test_invalid_input_error():
    """Tests error for the invalid input when both iterations
    and delays are provided."""
    with pytest.raises(ValueError):
        load_images(scan_path_single, iterations=[0], delays=[0])


def test_iterations_type():
    """Tests different allowed object types of iterations"""
    iter_list = [0, 1]
    iter_array = np.arange(2)
    iter_slice = np.s_[0:2]
    data_list = load_images(scan_path_mirror, df_lut, iter_list)
    data_array = load_images(scan_path_mirror, df_lut, iter_array)
    data_slice = load_images(scan_path_mirror, df_lut, iter_slice)

    assert np.array_equal(data_list, data_array)
    assert np.array_equal(data_array, data_slice)
