"""This is a code that performs several tests for the SpecsScan core class functions
"""
import os

import numpy as np
import pytest
from pynxtools.dataconverter.convert import ValidationFailed

import specsscan
from specsanalyzer.core import create_fft_params
from specsscan import __version__
from specsscan import SpecsScan

package_dir = os.path.dirname(specsscan.__file__)
test_dir = package_dir + "/../tests/data/"
fft_filter_peaks = create_fft_params(amplitude=1, pos_x=82, pos_y=116, sigma_x=15, sigma_y=23)


def test_version():
    """Test if the package has the correct version string."""
    assert __version__ == "0.1.0"


def test_default_config():
    """Test if the default config can be loaded and test for one entry."""
    sps = SpecsScan(user_config={}, system_config={})
    assert isinstance(sps.config, dict)
    assert "spa_params" in sps.config.keys()
    assert sps.config["spa_params"]["apply_fft_filter"] is False


def test_conversion_2d():
    """Test the conversion of a single-image scan"""
    sps = SpecsScan(
        config=test_dir + "config.yaml",
        user_config={},
        system_config={},
    )
    res_xarray = sps.load_scan(
        scan=3610,
        path=test_dir,
    )
    assert res_xarray.dims == ("Angle", "Ekin")

    with pytest.raises(IndexError):
        res_xarray = sps.load_scan(
            scan=3610,
            path=test_dir,
            iterations=[0],
        )


def test_conversion_3d():
    """Test the conversion of a 3D scan"""
    sps = SpecsScan(
        config=test_dir + "config.yaml",
        user_config={},
        system_config={},
    )
    res_xarray = sps.load_scan(
        scan=4450,
        path=test_dir,
    )
    assert res_xarray.dims == ("Angle", "Ekin", "mirrorX")

    res_xarray2 = sps.load_scan(
        scan=4450,
        path=test_dir,
        iterations=[0],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        res_xarray.values,
        res_xarray2.values,
    )

    res_xarray2 = sps.load_scan(
        scan=4450,
        path=test_dir,
        iterations=np.s_[0:2],
    )
    np.testing.assert_allclose(res_xarray, res_xarray2)

    with pytest.raises(IndexError):
        sps.load_scan(
            scan=4450,
            iterations=range(1, 20),
            path=test_dir,
        )


def test_conversion_from_convert_dict():
    """Test the conversion without calib2d file, using passed conversion dictionary parameters"""
    sps = SpecsScan(
        config={},
        user_config={},
        system_config={},
    )
    with pytest.raises(ValueError):
        res_xarray = sps.load_scan(
            scan=4450,
            path=test_dir,
        )

    conversion_parameters = {
        "lens_mode": "WideAngleMode",
        "kinetic_energy": 21.9,
        "pass_energy": 30.0,
        "work_function": 4.558,
        "a_inner": 15.0,
        "da_matrix": np.array(
            [
                [0.70585613, 0.74383533, 0.7415424],
                [-0.00736453, 0.05832768, 0.14868587],
                [-0.00759583, -0.04533556, -0.09021117],
                [-0.00180035, 0.00814881, 0.01743308],
            ],
        ),
        "retardation_ratio": 0.5780666666666666,
        "source": "interpolated as 0.4386666666666681*WideAngleMode@0.55 + 0.5613333333333319*WideAngleMode@0.6",  # noqa
        "dims": ["Angle", "Ekin"],
        "e_shift": np.array([-0.05, 0.0, 0.05]),
        "de1": [0.0033],
        "e_range": [-0.066, 0.066],
        "a_range": [-15.0, 15.0],
        "pixel_size": 0.0258,
        "magnification": 4.54,
        "angle_offset_px": 0,
        "energy_offset_px": 0,
    }

    res_xarray = sps.load_scan(
        scan=4450,
        path=test_dir,
        conversion_parameters=conversion_parameters,
    )

    for key in conversion_parameters.keys():
        assert key in res_xarray.attrs["metadata"]["conversion_parameters"]


def test_checkscan():
    """Test the check_scan function"""
    sps = SpecsScan(
        config=test_dir + "config.yaml",
        user_config={},
        system_config={},
    )

    res_xarray = sps.check_scan(
        scan=4450,
        delays=[0],
        path=test_dir,
    )
    assert res_xarray.dims == ("Angle", "Ekin", "Iteration")

    with pytest.raises(IndexError):
        sps.check_scan(
            scan=4450,
            delays=range(1, 20),
            path=test_dir,
        )


def test_checkscan_2d_raises():
    """Test that the check_scan function raises if a single image is loaded"""
    sps = SpecsScan(
        config=test_dir + "config.yaml",
        user_config={},
        system_config={},
    )

    with pytest.raises(ValueError):
        sps.check_scan(
            scan=3610,
            delays=[0],
            path=test_dir,
        )


def test_process_sweep_scan():
    """Test the conversion of a sweep scan"""
    config = {
        "spa_params": {
            "ek_range_min": 0.07597844332538357,
            "ek_range_max": 0.8965456312395133,
            "ang_range_min": 0.16732026143790849,
            "ang_range_max": 0.8449673202614381,
            "Ang_Offset_px": 13,
            "rotation_angle": 2,
            "crop": True,
        },
    }
    sps = SpecsScan(
        config=config,
        user_config=package_dir + "/config/example_config_FHI.yaml",
        system_config={},
    )
    res_xarray = sps.load_scan(
        scan=6455,
        path=test_dir,
    )
    assert res_xarray.energy[0].values.item() == 20.953256232558136
    assert res_xarray.energy[-1].values.item() == 21.02424460465116
    assert (
        (res_xarray.sum(axis=0) - res_xarray.sum(axis=0).mean()) < 0.1 * res_xarray.sum(axis=0)
    ).all()


def test_crop_tool():
    """Test the crop tool"""
    sps = SpecsScan(
        config=test_dir + "config.yaml",
        user_config={},
        system_config={},
    )

    res_xarray = sps.load_scan(
        scan=3610,
        path=test_dir,
        crop=True,
    )

    assert res_xarray.Angle[0] == -18
    assert res_xarray.Angle[-1] == 17.859375
    # assert res_xarray.Angle[0] == -15
    # assert res_xarray.Angle[-1] == 14.8828125
    assert res_xarray.Ekin[0] == 18.69
    assert res_xarray.Ekin[-1] == 23.29656976744186

    res_xarray = sps.load_scan(
        scan=3610,
        path=test_dir,
        ek_range_min=0.1,
        ek_range_max=0.9,
        ang_range_min=0.1,
        ang_range_max=0.9,
        crop=True,
    )
    assert res_xarray.Angle[0] == -14.34375
    assert res_xarray.Angle[-1] == 14.203125
    # assert res_xarray.Angle[0] == -11.953125
    # assert res_xarray.Angle[-1] == 11.8359375
    assert res_xarray.Ekin[0] == 19.160058139534886
    assert res_xarray.Ekin[-1] == 22.826511627906974

    sps.crop_tool(
        ek_range_min=0.1,
        ek_range_max=0.9,
        ang_range_min=0.1,
        ang_range_max=0.9,
        apply=True,
    )

    res_xarray = sps.load_scan(
        scan=3610,
        path=test_dir,
    )
    assert res_xarray.Angle[0] == -14.34375
    assert res_xarray.Angle[-1] == 14.203125
    # assert res_xarray.Angle[0] == -11.953125
    # assert res_xarray.Angle[-1] == 11.8359375
    assert res_xarray.Ekin[0] == 19.160058139534886
    assert res_xarray.Ekin[-1] == 22.826511627906974


def test_fft_tool():
    """Test the fft tool"""

    sps = SpecsScan(
        config=test_dir + "config.yaml",
        user_config={},
        system_config={},
    )
    res_xarray = sps.load_scan(
        scan=3610,
        path=test_dir,
        apply_fft_filter=False,
    )

    np.testing.assert_almost_equal(res_xarray.data.sum(), 62145561928.15108, decimal=3)

    res_xarray = sps.load_scan(
        scan=3610,
        path=test_dir,
        fft_filter_peaks=fft_filter_peaks,
        apply_fft_filter=True,
    )
    np.testing.assert_almost_equal(res_xarray.data.sum(), 62197237155.50347, decimal=3)

    sps.fft_tool(
        amplitude=1,
        pos_x=82,
        pos_y=116,
        sigma_x=15,
        sigma_y=23,
        apply=True,
    )
    assert sps.config["spa_params"]["fft_filter_peaks"] == fft_filter_peaks
    assert sps.spa.config["fft_filter_peaks"] == fft_filter_peaks
    res_xarray = sps.load_scan(scan=3610, path=test_dir, apply_fft_filter=True)
    np.testing.assert_almost_equal(res_xarray.data.sum(), 62197237155.50347, decimal=3)


def test_conversion_and_save_to_nexus():
    """Test the conversion of a tilt scan and saving as NeXus"""
    config = {"nexus": {"input_files": [package_dir + "/config/NXmpes_arpes_config.json"]}}
    sps = SpecsScan(
        config=config,
        user_config=package_dir + "/config/example_config_FHI.yaml",
        system_config={},
    )

    res_xarray = sps.load_scan(
        scan=1496,
        path=test_dir,
        crop=True,
    )

    assert res_xarray.dims == ("angular0", "angular1", "energy")

    with pytest.raises(NameError):
        sps.save("result.tiff")
    sps.save("result.tiff", alias_dict={"X": "angular0", "Y": "angular1"})
    sps.save("result.h5")
    with pytest.raises(ValidationFailed):
        sps.save("result.nxs", fail=True)

    metadata = {}
    # General
    metadata["experiment_summary"] = "summary"
    metadata["entry_title"] = "title"
    metadata["experiment_title"] = "exp_title"

    metadata["instrument"] = {}
    # energy resolution
    metadata["instrument"]["energy_resolution"] = 150.0
    metadata["instrument"]["electronanalyser"] = {}
    metadata["instrument"]["electronanalyser"]["energy_resolution"] = 120
    metadata["instrument"]["electronanalyser"]["angular_resolution"] = 0.2
    metadata["instrument"]["electronanalyser"]["spatial_resolution"] = 0.5

    # probe beam
    metadata["instrument"]["beam"] = {}
    metadata["instrument"]["beam"]["probe"] = {}
    metadata["instrument"]["beam"]["probe"]["incident_energy"] = 21.7

    # sample
    metadata["sample"] = {}
    metadata["sample"]["name"] = "Name"

    metadata["scan_info"] = {}
    metadata["scan_info"]["trARPES:XGS600:PressureAC:P_RD"] = 2.5e-11
    metadata["scan_info"]["trARPES:Carving:TEMP_RBV"] = 70
    metadata["scan_info"]["trARPES:Sample:Measure"] = 0
    res_xarray = sps.load_scan(
        scan=1496,
        path=test_dir,
        crop=True,
        metadata=metadata,
        collect_metadata=True,
    )

    sps.save("result.nxs", fail=True)
