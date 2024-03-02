# pylint: disable=duplicate-code
"""This is a code that performs several tests for the convert functions
"""
import os

import numpy as np

import specsanalyzer
from specsanalyzer import SpecsAnalyzer
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer.convert import calculate_polynomial_coef_da
from specsanalyzer.convert import get_damatrix_fromcalib2d

package_dir = os.path.dirname(specsanalyzer.__file__)
test_dir = package_dir + "/../tests/data/"

# from specsanalyzer.convert import get_rr_da
# from specsanalyzer.convert import mcp_position_mm
# let's get all the functions to be tested


def test_da_matrix():  # pylint: disable=too-many-locals
    """Check the consistency of the da matrix and the
    da poly matrix with the Igor calculations"""

    ########################################
    # Load the IGOR txt Di_coeff values for comparison
    igor_data_path = os.fspath(f"{test_dir}/dataEPFL/R9132")

    # get the Da coefficients
    di_file_list = [f"{igor_data_path}/Da{i}_value.tsv" for i in np.arange(1, 8, 2)]
    igor_d_value_list = []
    for file in di_file_list:
        with open(file, encoding="utf-8") as f_handle:
            igor_d_value_list.append(np.loadtxt(f_handle, delimiter="\t"))
    igor_d_value_matrix = np.vstack(igor_d_value_list)

    # get the fitted polynomial coefficients
    di_coef_list = [f"{igor_data_path}/D{i}_coef.tsv" for i in np.arange(1, 8, 2)]
    igor_d_coef_list = []
    for file in di_coef_list:
        with open(file, encoding="utf-8") as f_handle:
            igor_d_coef_list.append(np.loadtxt(f_handle, delimiter="\t"))
    igor_d_coef_matrix = np.flip(np.vstack(igor_d_coef_list), axis=1)

    config_path = os.fspath(f"{test_dir}/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=config_path)
    config_dict = spa.config
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2

    # get the matrix_correction
    e_shift = np.array(config_dict["calib2d_dict"]["eShift"])
    a_inner, da_matrix = get_damatrix_fromcalib2d(  # pylint: disable=W0612
        lens_mode,
        kinetic_energy,
        pass_energy,
        work_function,
        config_dict,
    )
    # get the polynomial coefficent matrix
    da_poly_matrix = calculate_polynomial_coef_da(
        da_matrix,
        kinetic_energy,
        pass_energy,
        e_shift,
    )

    np.testing.assert_allclose(da_matrix, igor_d_value_matrix, rtol=1e-05)
    np.testing.assert_allclose(da_poly_matrix, igor_d_coef_matrix, rtol=1e-05)


def test_conversion_matrix():  # pylint:disable=too-many-locals
    """Check the consistency of the conversion matrix with the
    Igor calculations.
    """
    igor_data_path = os.fspath(f"{test_dir}/dataEPFL/R9132")
    config_path = os.fspath(f"{test_dir}/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=config_path)
    config_dict = spa.config
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2
    binning = 4

    # Jacobian_correction_reference
    jacobian_file_name = f"{igor_data_path}/Jacobian_Determinant.tsv"
    with open(jacobian_file_name, encoding="utf-8") as file:
        jacobian_reference = np.loadtxt(file, delimiter="\t").T

    # angle_correction_reference
    angular_correction_file_name = f"{igor_data_path}/Angular_Correction.tsv"
    with open(angular_correction_file_name, encoding="utf-8") as file:
        angular_correction_reference = np.loadtxt(file, delimiter="\t").T

    # e_correction
    e_correction_file_name = f"{igor_data_path}/E_Correction.tsv"
    with open(e_correction_file_name, encoding="utf-8") as file:
        e_correction_reference = np.loadtxt(file, delimiter="\t")

    # angular axis
    angle_axis_file_name = f"{igor_data_path}/Data9132_angle.tsv"
    with open(angle_axis_file_name, encoding="utf-8") as file:
        angle_axis_ref = np.loadtxt(file, delimiter="\t")

    # ek axis axis
    ek_axis_file_name = f"{igor_data_path}/Data9132_energy.tsv"
    with open(ek_axis_file_name, encoding="utf-8") as file:
        ek_axis_ref = np.loadtxt(file, delimiter="\t")

    # get the matrix_correction
    (
        ek_axis,
        angle_axis,
        angular_correction_matrix,
        e_correction,
        jacobian_determinant,
    ) = calculate_matrix_correction(
        lens_mode,
        kinetic_energy,
        pass_energy,
        work_function,
        binning,
        config_dict,
    )

    np.testing.assert_allclose(
        angular_correction_matrix,
        angular_correction_reference,
        rtol=2e-02,
    )

    np.testing.assert_allclose(
        jacobian_determinant,
        jacobian_reference,
        rtol=1e-04,
    )

    np.testing.assert_allclose(
        e_correction,
        e_correction_reference,
        rtol=1e-04,
    )

    np.testing.assert_allclose(
        angle_axis,
        angle_axis_ref,
        rtol=1e-04,
    )

    np.testing.assert_allclose(
        ek_axis,
        ek_axis_ref,
        rtol=1e-04,
    )


def test_conversion():
    "Test if the conversion pipeline gives the same result as the Igor procedures"

    # get the raw data
    raw_image_name = os.fspath(
        f"{test_dir}/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name, encoding="utf-8") as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    # get the reference data
    reference_image_name = os.fspath(
        f"{test_dir}/dataEPFL/R9132/Data9132_IGOR_corrected.tsv",
    )
    with open(reference_image_name, encoding="utf-8") as file:
        reference = np.loadtxt(file, delimiter="\t")

    config_path = os.fspath(f"{test_dir}/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=config_path)
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        apply_fft_filter=False,
    )
    # Calculate the average intensity of the image, neglect the noisy parts
    # normalize to unit amplitude
    python_data = converted.data
    igor_data = reference
    python_data /= igor_data.max()
    igor_data /= igor_data.max()

    np.testing.assert_allclose(python_data, igor_data, atol=5e-5)


def test_recycling():
    """Test function for checking that the class correctly re-uses the
    precalculated parameters
    """
    # get the raw data
    raw_image_name = os.fspath(
        f"{test_dir}/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name, encoding="utf-8") as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    config_path = os.fspath(f"{test_dir}/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=config_path)
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2

    spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        apply_fft_filter=False,
    )

    assert spa.correction_matrix_dict["old_matrix_check"] is False

    spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        apply_fft_filter=False,
    )

    assert spa.correction_matrix_dict["old_matrix_check"] is True


def test_cropping():
    """Test function for checking that cropping parameters are correctly appield"""
    # get the raw data
    raw_image_name = os.fspath(
        f"{test_dir}/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name, encoding="utf-8") as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    config_path = os.fspath(f"{test_dir}/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=config_path)
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        crop=True,
    )
    assert converted.Angle[0] == -18
    assert converted.Angle[-1] == 17.859375
    assert converted.Ekin[0] == 32.69
    assert converted.Ekin[-1] == 37.296569767441866

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        ek_range_min=0.1,
        ek_range_max=0.9,
        ang_range_min=0.1,
        ang_range_max=0.9,
        crop=True,
    )
    assert converted.Angle[0] == -14.34375
    assert converted.Angle[-1] == 14.203125
    assert converted.Ekin[0] == 33.16005813953488
    assert converted.Ekin[-1] == 36.82651162790698

    spa.crop_tool(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        ek_range_min=0.1,
        ek_range_max=0.9,
        ang_range_min=0.1,
        ang_range_max=0.9,
        apply=True,
    )

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        crop=True,
    )

    assert converted.Angle[0] == -14.34375
    assert converted.Angle[-1] == 14.203125
    assert converted.Ekin[0] == 33.16005813953488
    assert converted.Ekin[-1] == 36.82651162790698

    spa.crop_tool(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=45.0,
        pass_energy=pass_energy,
        work_function=work_function,
        ek_range_min=0.2,
        ek_range_max=0.8,
        ang_range_min=0.2,
        ang_range_max=0.8,
        apply=True,
    )

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=50.0,
        pass_energy=pass_energy,
        work_function=work_function,
        crop=True,
    )

    assert converted.Angle[0] == -10.828125
    assert converted.Angle[-1] == 10.6875
    assert converted.Ekin[0] == 48.616686046511624
    assert converted.Ekin[-1] == 51.36988372093023

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        crop=True,
    )

    assert converted.Angle[0] == -14.34375
    assert converted.Angle[-1] == 14.203125
    assert converted.Ekin[0] == 33.16005813953488
    assert converted.Ekin[-1] == 36.82651162790698
