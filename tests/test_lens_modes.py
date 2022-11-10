"""This is a code that performs several tests for the convert functions
"""
import os

import numpy as np
import pytest

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
test_io = []
for mode in lensmodes_angle + lensmodes_space:
    test_io.append((mode, True))


@pytest.mark.parametrize("lens_mode,expected", test_io)
def test_lens_modes(lens_mode, expected):
    """Test that all the supported lens modes run without error"""
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

    try:
        converted = spa.convert_image(  # noqa: F841 # pylint: disable=W0612
            raw_img=tsv_data,
            lens_mode=lens_mode,
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            apply_fft_filter=False,
        )
        test_result = True
    except KeyError:
        test_result = False
    assert test_result == expected


# test_io_2=(error_lens_mode,expected_out)
# @pytest.mark.parametrize("error_lens_mode,expected_out", test_io_2)
def test_lens_raise():
    error_lens_mode = "WideAngleModel"
    expected_out = "convert_image: unsupported lens mode: WideAngleModel"

    """Test if program raises suitable errors"""
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

    try:
        converted = spa.convert_image(  # noqa: F841 # pylint: disable=W0612
            raw_img=tsv_data,
            lens_mode=error_lens_mode,
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            apply_fft_filter=False,
        )
        test_result = True
    except ValueError as error:
        print("Found value error: ")
        print(str(error))
        test_result = str(error)
    assert test_result == expected_out
