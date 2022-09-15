"""This is a code that performs several tests for the convert functions
"""
import os

import numpy as np

from specsanalyzer import SpecsAnalyzer
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer.convert import calculate_polynomial_coef_da
from specsanalyzer.convert import get_damatrix_fromcalib2d

test_dir = os.path.dirname(__file__)

# from specsanalyzer.convert import get_rr_da
# from specsanalyzer.convert import mcp_position_mm
# let's get all the functions to be tested


def test_da_matrix():
    """Check the consistency of the da matrix and the
    da poly matrix with the Igor calculations"""

    ########################################
    # Load the IGOR txt Di_coeff values for comparison
    igordatapath = os.fspath(f"{test_dir}/data/dataEPFL/R9132")

    # get the Da coefficients
    Di_file_list = [
        f"{igordatapath}/Da{i}_value.tsv" for i in np.arange(1, 8, 2)
    ]
    igor_D_value_list = []
    for file in Di_file_list:
        with open(file) as f:
            igor_D_value_list.append(np.loadtxt(f, delimiter="\t"))
    igor_D_value_matrix = np.vstack(igor_D_value_list)

    # get the fitted polynomial coefficients
    Di_coef_list = [
        f"{igordatapath}/D{i}_coef.tsv" for i in np.arange(1, 8, 2)
    ]
    igor_D_coef_list = []
    for file in Di_coef_list:
        with open(file) as f:
            igor_D_coef_list.append(np.loadtxt(f, delimiter="\t"))
    igor_D_coef_matrix = np.flip(np.vstack(igor_D_coef_list), axis=1)

    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    config_dict = spa.config
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2

    eshift = np.array(config_dict["calib2d_dict"]["eShift"])
    a_inner, damatrix = get_damatrix_fromcalib2d(
        lens_mode,
        kinetic_energy,
        pass_energy,
        work_function,
        config_dict,
    )
    # get the polynomial coefficent matrix
    dapolymatrix = calculate_polynomial_coef_da(
        damatrix,
        kinetic_energy,
        pass_energy,
        eshift,
    )

    np.testing.assert_allclose(damatrix, igor_D_value_matrix, rtol=1e-05)
    np.testing.assert_allclose(dapolymatrix, igor_D_coef_matrix, rtol=1e-05)


def test_conversion_matrix():
    """Check the consistency of the conversion matrix with the
    Igor calculations.
    """
    igordatapath = os.fspath(f"{test_dir}/data/dataEPFL/R9132")
    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    config_dict = spa.config
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2
    binning = 4

    # Jacobian_correction_reference
    jacobian_file_name = f"{igordatapath}/Jacobian_Determinant.tsv"
    with open(jacobian_file_name) as file:
        jacobian_reference = np.loadtxt(file, delimiter="\t").T

    # angle_correction_reference
    angular_correction_file_name = f"{igordatapath}/Angular_Correction.tsv"
    with open(angular_correction_file_name) as file:
        angular_correction_reference = np.loadtxt(file, delimiter="\t").T

    # e_correction
    E_correction_file_name = f"{igordatapath}/E_Correction.tsv"
    with open(E_correction_file_name) as file:
        e_correction_reference = np.loadtxt(file, delimiter="\t")

    # TODO: Seems to be wrong references...

    # angular axis
    angle_axis_file_name = f"{igordatapath}/Data9132_angle.tsv"
    with open(angle_axis_file_name) as file:
        angle_axis_ref = np.loadtxt(file, delimiter="\t")

    # ek axis axis
    ek_axis_file_name = f"{igordatapath}/Data9132_energy.tsv"
    with open(ek_axis_file_name) as file:
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
        f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name) as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    # get the reference data
    reference_image_name = os.fspath(
        f"{test_dir}/data/dataEPFL/R9132/Data9132_IGOR_corrected.tsv",
    )
    with open(reference_image_name) as file:
        reference = np.loadtxt(file, delimiter="\t")

    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
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

    # TODO Does not work yet... Not sure how you produced the reference?
    np.testing.assert_allclose(python_data, igor_data, atol=5e-5)


def test_recycling():
    """Test function for chceking that the class correctly re-uses the
    precalculated parameters
    """
    # get the raw data
    raw_image_name = os.fspath(
        f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name) as file:
        tsv_data = np.loadtxt(file, delimiter="\t")

    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
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

    converted = spa.convert_image(
        raw_img=tsv_data,
        lens_mode=lens_mode,
        kinetic_energy=kinetic_energy,
        pass_energy=pass_energy,
        work_function=work_function,
        apply_fft_filter=False,
    )
    converted
    currentdict = spa.correction_matrix_dict

    testresult = currentdict["old_matrix_check"]
    assert testresult
