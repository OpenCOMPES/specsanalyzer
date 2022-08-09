import os

import specsanalyzer

from specsscan import __version__
from specsscan import SpecsScan

package_dir = os.path.dirname(specsanalyzer.__file__)

def test_version():
    assert __version__ == "0.1.0"


def test_default_config():
    sps = SpecsScan(config={'spa_params':{'calib2d_file' : f"{package_dir}/../tests/data/config/phoibos150.calib2d"}})
    assert isinstance(sps._config, dict)
    assert 'spa_params' in sps._config.keys()
    assert sps._config['spa_params']['apply_fft_filter'] == False
