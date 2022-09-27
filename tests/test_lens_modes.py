"""This is a code that performs several tests for the convert functions
"""
import os
from typing import Concatenate

import numpy as np

from specsanalyzer import SpecsAnalyzer

test_dir = os.path.dirname(__file__)
# noqa: EF841

def test_lens_modes():  
    """Test that all the supported lens modes run without error
    """
    raw_image_name = os.fspath(
        f"{test_dir}/data/dataEPFL/R9132/Data9132_RAWDATA.tsv",
    )
    with open(raw_image_name) as file:
        tsv_data = np.loadtxt(file, delimiter="\t")
    
    configpath = os.fspath(f"{test_dir}/data/dataEPFL/config/config.yaml")
    spa = SpecsAnalyzer(config=configpath)
    kinetic_energy = 35.000000
    pass_energy = 35.000000
    work_function = 4.2
    lensmodes_angle = [
        "WideAngleMode",
        "LowAngularDispersion",
        "MediumAngularDispersion",
        "HighAngularDispersion",
        "WideAngleMode",
        "SuperWideAngleMode"
    ]
    lensmodes_space = [
        "LargeArea",
        "MediumArea",
        "SmallArea",
        "SmallArea2",
        "HighMagnification2",
        "HighMagnification",
        "MediumMagnification",
        "LowMagnification"
    ]
    lens_mode_list = lensmodes_angle+lensmodes_space    
    test_result = False
    for lens_mode in lens_mode_list:
        try: 
            converted = spa.convert_image(  # noqa: EF841
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
    assert test_result