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


# # test_io_2=(error_lens_mode,expected_out)
# # @pytest.mark.parametrize("error_lens_mode,expected_out", test_io_2)
# def test_lens_raise():
#     error_lens_mode = "WideAngleModel"
#     expected_out = "convert_image: unsupported lens mode: WideAngleModel"

#     """Test if program raises suitable errors"""
#     raw_image_name = os.fspath(
#         f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
#     )
#     with open(raw_image_name) as file:  # pylint: disable=W1514
#         tsv_data = np.loadtxt(file, delimiter="\t")

#     configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
#     spa = SpecsAnalyzer(config=configpath)
#     kinetic_energy = 35.000000
#     pass_energy = 35.000000
#     work_function = 4.2

#     try:
#         converted = spa.convert_image(  # noqa: F841 # pylint: disable=W0612
#             raw_img=tsv_data,
#             lens_mode=error_lens_mode,
#             kinetic_energy=kinetic_energy,
#             pass_energy=pass_energy,
#             work_function=work_function,
#             apply_fft_filter=False,
#         )
#         test_result = True
#     except ValueError as error:
#         print("Found value error: ")
#         print(str(error))
#         test_result = str(error)
#     assert test_result == expected_out


# def test_lens_traceback():

#     expected_out = (
#         "The supported modes were not found in the calib2d dictionary"
#     )

#     """Test if program raises suitable errors"""
#     raw_image_name = os.fspath(
#         f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
#     )
#     with open(raw_image_name) as file:  # pylint: disable=W1514
#         tsv_data = np.loadtxt(file, delimiter="\t")

#     configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
#     spa = SpecsAnalyzer(config=configpath)
#     kinetic_energy = 35.000000
#     pass_energy = 35.000000
#     work_function = 4.2

#     # let's delibertaly remove the keys from the class config dictionary#
#     spa.config["calib2d_dict"].pop("supported_angle_modes")
#     spa.config["calib2d_dict"].pop("supported_space_modes")
#     #################################################################

#     try:

#         converted = spa.convert_image(  # noqa: F841 # pylint: disable=W0612
#             raw_img=tsv_data,
#             lens_mode="WideAngleMode",
#             kinetic_energy=kinetic_energy,
#             pass_energy=pass_energy,
#             work_function=work_function,
#             apply_fft_filter=False,
#         )

#         test_result = True
#     except KeyError as error:
#         print("Found key error: ")
#         print(str(error))
#         test_result = str(error)[1:-1]  # this removes the '
#     print(test_result, expected_out)
#     assert test_result == expected_out


def test_lens_raise():
    """Test if the conversion raises an error for a wrong or 
    missing lens mode."""
    with pytest.raises(ValueError):
        error_lens_mode = "WideAngleModel"

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

   
        converted = spa.convert_image(  # noqa: F841 # pylint: disable=W0612
            raw_img=tsv_data,
            lens_mode=error_lens_mode,
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            apply_fft_filter=False,
        )
   

def test_supported_modes_raise():
    """Test if the conversion raises an error for a missing
    entry in the config dict with the supported modes."""
    
    with pytest.raises(KeyError):
        error_lens_mode = "WideAngleMode"

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

        spa.config["calib2d_dict"].pop("supported_angle_modes")
        spa.config["calib2d_dict"].pop("supported_space_modes")
    
        converted = spa.convert_image(  # noqa: F841 # pylint: disable=W0612
            raw_img=tsv_data,
            lens_mode=error_lens_mode,
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            apply_fft_filter=False,
        )