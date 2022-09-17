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
    sps = SpecsScan()
    assert isinstance(sps.config, dict)
    assert "spa_params" in sps.config.keys()
    assert sps.config["spa_params"]["apply_fft_filter"] is False
