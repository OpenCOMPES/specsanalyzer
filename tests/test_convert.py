"""This is a code that performs several tests for the convert functions
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
# let's get all the functions to be tested
from specsanalyzer.convert import get_damatrix_fromcalib2d
from specsanalyzer.convert import get_rr_da
from specsanalyzer.convert import calculate_polynomial_coef_da
from specsanalyzer.convert import mcp_position_mm
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer import SpecsAnalyzer
import os

def test_da_matrix():

    # get the raw data
    raw_image_name=os.fspath('./tests/data/dataEPFL/R9132/Data9132_RAWDATA.tsv')
    with open(raw_image_name) as file:
        tsv_data = np.loadtxt(file, delimiter="\t")
    ########################################

    # Load the IGOR txt Di_coeff values for comparison
    igordatapath = os.fspath("./tests/data/dataEPFL/R9132")
    igordatapath_content = os.listdir(igordatapath)

    # get the Da coefficients
    Di_value_list = [i for i in igordatapath_content if "_value.tsv" in i]
    igor_D_value_list = []
    for i, name in enumerate(sorted(Di_value_list)):
        tmp_name = os.path.join(igordatapath, name)
        with open(tmp_name) as file:
            igor_D_value_list.append(np.loadtxt(file, delimiter="\t"))
    igor_D_value_matrix = np.vstack(igor_D_value_list)

    # get the fitted polynomial coefficients
    Di_coef_list = [i for i in igordatapath_content if "_coef" in i]
    igor_D_coef_list = []
    for i, name in enumerate(sorted(Di_coef_list)):
        tmp_name = os.path.join(igordatapath, name)
        with open(tmp_name) as file:
            igor_D_coef_list.append(np.loadtxt(file, delimiter="\t"))
    igor_D_coef_matrix = np.flip(np.vstack(igor_D_coef_list), axis=1)
  
    # Jacobian_correction_reference
    jname = [i for i in igordatapath_content if "Jacobian" in i][0]
    with open(os.path.join(igordatapath, jname)) as file:
        jacobian_reference = np.loadtxt(file, delimiter="\t").T

    # angle_correction_reference
    jname = [i for i in igordatapath_content if "Angular_Correction" in i][0]
    jname
    with open(os.path.join(igordatapath, jname)) as file:
       angle_correction_reference = np.loadtxt(file, delimiter="\t").T
    # e_correction
    jname = [i for i in igordatapath_content if "E_Correction" in i][0]
    jname
    with open(os.path.join(igordatapath, jname)) as file:
        e_correction_reference = np.loadtxt(file, delimiter="\t")

    configpath = os.fspath("./tests/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    config_dict = spa.config
    lens_mode = "WideAngleMode"
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2
    binning = 4

    eshift = np.array(config_dict["calib2d_dict"]["eShift"])
    aInner, damatrix = get_damatrix_fromcalib2d(
        lens_mode,
        kinetic_energy,
        pass_energy,
        work_function,
        config_dict,
    )
    # get the polynomial coefficent matrix
    dapolymatrix = calculate_polynomial_coef_da(
        damatrix, kinetic_energy, pass_energy, eshift
    )
    
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
    
   
    
    
    np.testing.assert_allclose(damatrix, igor_D_value_matrix, rtol=1e-05)       
    np.testing.assert_allclose(dapolymatrix, igor_D_coef_matrix, rtol=1e-05)
    np.testing.assert_allclose(jacobian_determinant, jacobian_reference, rtol=1e-04)
    np.testing.assert_allclose(e_correction, e_correction_reference, rtol=1e-04)
    
    