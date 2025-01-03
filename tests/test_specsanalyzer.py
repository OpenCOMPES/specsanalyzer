"""This is a code that performs several tests for the SpecsAnalyzer core class functions"""
import importlib.metadata

from specsanalyzer import __version__
from specsanalyzer import SpecsAnalyzer


def test_version():
    """Test if the package has the correct version string."""
    assert __version__ == importlib.metadata.version("specsanalyzer")


def test_default_config():
    """Test if the default config can be loaded and test for one entry."""
    spa = SpecsAnalyzer(user_config={}, system_config={})
    assert isinstance(spa.config, dict)
    assert spa.config["apply_fft_filter"] is False
