"""This is a code that performs several tests for the settings loader."""
import os
import tempfile
from pathlib import Path

import pytest

from specsanalyzer.config import complete_dictionary
from specsanalyzer.config import load_config
from specsanalyzer.config import parse_config
from specsanalyzer.config import save_config

test_dir = os.path.dirname(__file__)
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
        f"{test_dir}/data/config/config.json",
    )
    config_yaml = load_config(
        f"{test_dir}/data/config/config.yaml",
    )
    assert config_json == config_yaml


def test_load_config_raise():
    """Test if the config loader raises an error for a wrong file type."""
    with pytest.raises(TypeError):
        load_config(f"{test_dir}/../README.md")


def test_complete_dictionary():
    """Test the merging of a config and a default config dict"""
    dict1 = {"key1": 1, "key2": 2, "nesteddict": {"key4": 4}}
    dict2 = {"key1": 2, "key3": 3, "nesteddict": {"key5": 5}}
    dict3 = complete_dictionary(dictionary=dict1, base_dictionary=dict2)
    assert isinstance(dict3, dict)
    for key in ["key1", "key2", "key3", "nesteddict"]:
        assert key in dict3
    for key in ["key4", "key5"]:
        assert key in dict3["nesteddict"]
    assert dict3["key1"] == 1


def test_save_dict():
    """Test the config saver for a dict."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        for ext in ["yaml", "json"]:
            filename = tmpdirname + "/.sed_config." + ext
            config_dict = {"test_entry": True}
            save_config(config_dict, filename)
            assert Path(filename).exists()
            config = load_config(filename)
            assert config == config_dict
            config_dict = {"test_entry2": False}
            save_config(config_dict, filename)
            config = load_config(filename)
            assert {"test_entry", "test_entry2"}.issubset(config.keys())
            config_dict = {"test_entry2": False}
            save_config(config_dict, filename, overwrite=True)
            config = load_config(filename)
            assert "test_entry" not in config.keys()
