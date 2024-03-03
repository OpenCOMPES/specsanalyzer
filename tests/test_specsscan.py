"""This is a code that performs several tests for the SpecsScan core class functions
"""
import os

import numpy as np
import pytest

import specsscan
from specsscan import __version__
from specsscan import SpecsScan

package_dir = os.path.dirname(specsscan.__file__)
test_dir = package_dir + "/../tests/data/"


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
    assert res_xarray.sum(axis=(0, 1, 2)) != res_xarray2.sum(axis=(0, 1, 2))

    res_xarray2 = sps.load_scan(
        scan=4450,
        path=test_dir,
        iterations=np.s_[0:2],
    )
    assert res_xarray.sum(axis=(0, 1, 2)) == res_xarray2.sum(axis=(0, 1, 2))

    with pytest.raises(IndexError):
        sps.check_scan(
            scan=4450,
            delays=range(1, 20),
            path=test_dir,
        )


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
