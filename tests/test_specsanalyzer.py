import os

import specsanalyzer
from specsanalyzer import __version__
from specsanalyzer import SpecsAnalyzer

package_dir = os.path.dirname(specsanalyzer.__file__)


def test_version():
    assert __version__ == "0.1.0"


def test_default_config():
    spa = SpecsAnalyzer()
    assert isinstance(spa._config, dict)
    assert spa._config["apply_fft_filter"] is False
