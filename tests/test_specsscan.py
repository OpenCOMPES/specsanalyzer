from specsscan import __version__
from specsscan import SpecsScan


def test_version():
    assert __version__ == "0.1.0"


def test_default_config():
    sps = SpecsScan()
    assert isinstance(sps._config, dict)
    assert "spa_params" in sps._config.keys()
    assert sps._config["spa_params"]["apply_fft_filter"] is False
