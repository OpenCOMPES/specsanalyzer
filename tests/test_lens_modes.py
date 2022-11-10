"""This is a code that performs several tests for the convert functions
"""
import os

import numpy as np
import pytest
import xarray as xr

from specsanalyzer import SpecsAnalyzer

test_dir = os.path.dirname(__file__)
# noqa: EF841

lensmodes_angle = [
    "WideAngleMode",
    "LowAngularDispersion",
    "MediumAngularDispersion",
    "HighAngularDispersion",
    "WideAngleMode",
    "SuperWideAngleMode",
]
lensmodes_space = [
    "LargeArea",
    "MediumArea",
    "SmallArea",
    "SmallArea2",
    "HighMagnification2",
    "HighMagnification",
    "MediumMagnification",
    "LowMagnification",
]


@pytest.mark.parametrize("lens_mode", lensmodes_angle)
def test_lens_modes_angle(lens_mode):
    """Test that all the supported lens modes run without error"""
    raw_image_name = os.fspath(
        f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name, encoding="utf-8") as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    kinetic_energy = np.random.uniform(20, 50)
    pass_energy = np.random.uniform(20, 50)
    work_function = 4.2

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        apply_fft_filter=False,
    )

    assert isinstance(converted, xr.DataArray)
    assert "Angle" in converted.dims
    assert "Ekin" in converted.dims


@pytest.mark.parametrize("lens_mode", lensmodes_space)
def test_lens_modes_space(lens_mode):
    """Test that all the supported lens modes run without error"""
    raw_image_name = os.fspath(
        f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name, encoding="utf-8") as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    kinetic_energy = np.random.uniform(20, 50)
    pass_energy = np.random.uniform(20, 50)
    work_function = 4.2

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        apply_fft_filter=False,
    )

    assert isinstance(converted, xr.DataArray)
    assert "Position" in converted.dims
    assert "Ekin" in converted.dims


def test_lens_raise():
    """Test if program raises the correct error on unsupported lens mode."""
    raw_image_name = os.fspath(
        f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name) as file:  # pylint: disable=W1514
        tsv_data = np.loadtxt(file, delimiter="\t")

    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2

    with pytest.raises(ValueError):
        spa.convert_image(
            raw_img=tsv_data,
            lens_mode="WideAngleModel",
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            apply_fft_filter=False,
        )

    spa.config["calib2d_dict"].pop("supported_angle_modes")
    spa.config["calib2d_dict"].pop("supported_space_modes")

    with pytest.raises(KeyError):
        spa.convert_image(
            raw_img=tsv_data,
            lens_mode="WideAngleModel",
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            apply_fft_filter=False,
        )
