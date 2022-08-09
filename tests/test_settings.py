from pathlib import Path

import pytest

from specsscan.settings import insert_default_config
from specsscan.settings import load_config
from specsscan.settings import parse_config

default_config_keys = ['spa_params', 'data_path']
default_spa_config_keys = ['apply_fft_filter']

def test_default_config():
    config = parse_config()
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()
    assert isinstance(config['spa_params'], dict)
    for key in default_spa_config_keys:
        assert key in config['spa_params'].keys()

def test_load_dict():
    config_dict = {'test_entry': True}
    config = parse_config(config_dict)
    assert isinstance(config, dict)
    for key in default_config_keys:
        assert key in config.keys()
    assert config['test_entry'] == True

def test_load_config():
    config_json = load_config(Path("specsscan/tests/data/config.json"))
    config_yaml = load_config(Path("specsscan/tests/data/config.yaml"))
    assert config_json == config_yaml

def test_load_config_raise():
    with pytest.raises(TypeError):
        load_config("specsscan/requirements.txt")

def test_insert_default_config():
    dict1 = {'key1': 1, 'key2': 2, 'nesteddict':{'key4': 4}}
    dict2 = {'key1': 2, 'key3': 3, 'nesteddict':{'key5': 5}}
    dict3=insert_default_config(config=dict1, default_config=dict2)
    assert isinstance(dict3, dict)
    for key in ['key1', 'key2', 'key3', 'nesteddict']:
        assert key in dict3.keys()
    for key in ['key4', 'key5']:
        assert key in dict3['nesteddict'].keys()
    assert dict3['key1'] == 1
