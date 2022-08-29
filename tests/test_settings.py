"""This is a code that performs several tests for the settings loader.

"""
import os
import pytest

import specsanalyzer.settings
from specsanalyzer.settings import insert_default_config
from specsanalyzer.settings import load_config
from specsanalyzer.settings import parse_config

package_dir = os.path.dirname(specsanalyzer.__file__)
default_config_keys = [
    "calib2d_file",
    "nx_pixel",
    "ny_pixel",
    "pixel_size",
    "binning",
    "magnification",
    "apply_fft_filter",
]


def test_default_config():
    """Test the config loader for the default config."""
    config = parse_config()
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()


def test_load_dict():
    """Test the config loader for a dict."""
    config_dict = {"test_entry": True}
    config = parse_config(config_dict)
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()
    assert config["test_entry"] is True


def test_load_config():
    """Test if the config loader can handle json and yaml files."""
    config_json = load_config(
        f"{package_dir}/../tests/data/config/config.json",
    )
    config_yaml = load_config(
        f"{package_dir}/../tests/data/config/config.yaml",
    )
    assert config_json == config_yaml


def test_load_config_raise():
    """Test if the config loader raises an error for a wrong file type."""
    with pytest.raises(TypeError):
        load_config(f"{package_dir}/../requirements.txt")


def test_insert_default_config():
    """Test the merging of a config and a default config dict"""
    dict1 = {"key1": 1, "key2": 2, "nesteddict": {"key4": 4}}
    dict2 = {"key1": 2, "key3": 3, "nesteddict": {"key5": 5}}
    dict3 = insert_default_config(config=dict1, default_config=dict2)
    assert isinstance(dict3, dict)
    for key in ["key1", "key2", "key3", "nesteddict"]:
        assert key in dict3
    for key in ["key4", "key5"]:
        assert key in dict3["nesteddict"]
    assert dict3["key1"] == 1
