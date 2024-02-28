"""This is a code that performs several tests for the SpecsScan
core class functions
"""
import os

import specsscan
from specsscan import __version__
from specsscan import SpecsScan

package_dir = os.path.dirname(specsscan.__file__)


def test_version():
    """Test if the package has the correct version string."""
    assert __version__ == "0.1.0"


def test_default_config():
    """Test if the default config can be loaded and test for one entry."""
    sps = SpecsScan(user_config={}, system_config={})
    assert isinstance(sps.config, dict)
    assert "spa_params" in sps.config.keys()
    assert sps.config["spa_params"]["apply_fft_filter"] is False


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
        path=package_dir + "/../tests/data/",
    )
    assert res_xarray.energy[0].values.item() == 20.953256232558136
    assert res_xarray.energy[-1].values.item() == 21.02424460465116
    assert (
        (res_xarray.sum(axis=0) - res_xarray.sum(axis=0).mean()) < 0.1 * res_xarray.sum(axis=0)
    ).all()
