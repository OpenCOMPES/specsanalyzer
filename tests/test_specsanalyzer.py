"""This is a code that performs several tests for the SpecsAnalyzer
core class functions
"""
import os

import specsanalyzer
from specsanalyzer import __version__
from specsanalyzer import SpecsAnalyzer

package_dir = os.path.dirname(specsanalyzer.__file__)


def test_version():
    """Test if the package has the correct version string."""
    assert __version__ == "0.1.0"


def test_default_config():
    """Test if the default config can be loaded and test for one entry."""
    spa = SpecsAnalyzer()
    assert isinstance(spa.config, dict)
    assert spa.config["apply_fft_filter"] is False
