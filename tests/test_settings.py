from pathlib import Path

import pytest

from specsanalyzer.settings import insert_default_config
from specsanalyzer.settings import load_config
from specsanalyzer.settings import parse_config

default_config_keys = ['calib2d_file', 'nx_pixel', 'ny_pixel', 'pixel_size', 'magnification', 'apply_fft_filter']

def test_default_config():
    config = parse_config()
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()

def test_load_dict():
    config_dict = {'test_entry': True}
    config = parse_config(config_dict)
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()
    assert config['test_entry'] == True

def test_load_config():
    config_json = load_config(Path("specsanalyzer/tests/data/config/config.json"))
    config_yaml = load_config(Path("specsanalyzer/tests/data/config/config.yaml"))
    assert config_json == config_yaml

def test_load_config_raise():
    with pytest.raises(TypeError):
        load_config("specsanalyzer/requirements.txt")

def test_insert_default_config():
    dict1 = {'key1': 1, 'key2':2, 'nesteddict':{'key4':4}}
    dict2 = {'key1': 2, 'key3':3, 'nesteddict':{'key5':5}}
    dict3=insert_default_config(config=dict1, default_config=dict2)
    assert isinstance(dict3, dict)
    for key in ['key1', 'key2', 'key3', 'nesteddict']:
        assert key in dict3.keys()
    for key in ['key4', 'key5']:
        assert key in dict3['nesteddict'].keys()
    assert dict3['key1'] == 1
