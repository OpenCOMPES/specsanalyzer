"""This is a code that performs several tests for the settings loader."""
import os
import tempfile
from pathlib import Path

import pytest

from specsanalyzer.config import complete_dictionary
from specsanalyzer.config import load_config
from specsanalyzer.config import parse_config
from specsanalyzer.config import read_env_var
from specsanalyzer.config import save_config
from specsanalyzer.config import save_env_var

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


@pytest.fixture
def mock_env_file(tmp_path, monkeypatch):
    """Mock the .env file for testing"""
    monkeypatch.setattr("specsanalyzer.config.USER_CONFIG_PATH", tmp_path)
    yield tmp_path


def test_env_var_read_write(mock_env_file):  # noqa: ARG001
    """Test reading and writing environment variables."""
    # Test writing a new variable
    save_env_var("TEST_VAR", "test_value")
    assert read_env_var("TEST_VAR") == "test_value"

    # Test writing multiple variables
    save_env_var("TEST_VAR2", "test_value2")
    assert read_env_var("TEST_VAR") == "test_value"
    assert read_env_var("TEST_VAR2") == "test_value2"

    # Test overwriting an existing variable
    save_env_var("TEST_VAR", "new_value")
    assert read_env_var("TEST_VAR") == "new_value"
    assert read_env_var("TEST_VAR2") == "test_value2"  # Other variables unchanged

    # Test reading non-existent variable
    assert read_env_var("NON_EXISTENT_VAR") is None


def test_env_var_read_no_file(mock_env_file):  # noqa: ARG001
    """Test reading environment variables when .env file doesn't exist."""
    # Test reading from non-existent file
    assert read_env_var("TEST_VAR") is None


def test_env_var_special_characters(mock_env_file):  # noqa: ARG001
    """Test reading and writing environment variables with special characters."""
    test_cases = {
        "TEST_URL": "http://example.com/path?query=value",
        "TEST_PATH": "/path/to/something/with/spaces and special=chars",
        "TEST_QUOTES": "value with 'single' and \"double\" quotes",
    }

    for var_name, value in test_cases.items():
        save_env_var(var_name, value)
        assert read_env_var(var_name) == value


def test_env_var_precedence(mock_env_file, tmp_path, monkeypatch):  # noqa: ARG001
    """Test that environment variables are read in correct order of precedence"""
    # Create local .env directory if it doesn't exist
    local_env_dir = tmp_path / "local"
    local_env_dir.mkdir(exist_ok=True)
    system_env_dir = tmp_path / "system"
    system_env_dir.mkdir(exist_ok=True)
    monkeypatch.setattr("specsanalyzer.config.ENV_DIR", local_env_dir / ".env")
    monkeypatch.setattr("specsanalyzer.config.SYSTEM_CONFIG_PATH", system_env_dir)

    # Set up test values in different locations
    os.environ["TEST_VAR"] = "os_value"

    # Save to system config first (4th precedence)
    with open(system_env_dir / ".env", "w") as f:
        f.write("TEST_VAR=system_value\n")

    # Save to user config first (3rd precedence)
    save_env_var("TEST_VAR", "user_value")

    # Create local .env file (2nd precedence)
    with open(local_env_dir / ".env", "w") as f:
        f.write("TEST_VAR=local_value\n")

    assert read_env_var("TEST_VAR") == "os_value"

    # Remove from OS env to test other precedence levels
    monkeypatch.delenv("TEST_VAR", raising=False)
    assert read_env_var("TEST_VAR") == "local_value"

    # Remove local .env and should get user config value
    (local_env_dir / ".env").unlink()
    assert read_env_var("TEST_VAR") == "user_value"

    # Remove user config and should get system value
    (mock_env_file / ".env").unlink()
    assert read_env_var("TEST_VAR") == "system_value"

    # Remove system config and should get None
    (system_env_dir / ".env").unlink()
    assert read_env_var("TEST_VAR") is None


def test_env_var_save_and_load(mock_env_file, monkeypatch):  # noqa: ARG001
    """Test saving and loading environment variables"""
    # Clear any existing OS environment variables
    monkeypatch.delenv("TEST_VAR", raising=False)
    monkeypatch.delenv("OTHER_VAR", raising=False)

    # Save a variable
    save_env_var("TEST_VAR", "test_value")

    # Should be able to read it back
    assert read_env_var("TEST_VAR") == "test_value"

    # Save another variable - should preserve existing ones
    save_env_var("OTHER_VAR", "other_value")
    assert read_env_var("TEST_VAR") == "test_value"
    assert read_env_var("OTHER_VAR") == "other_value"


def test_env_var_not_found(mock_env_file):  # noqa: ARG001
    """Test behavior when environment variable is not found"""
    assert read_env_var("NONEXISTENT_VAR") is None


def test_env_file_format(mock_env_file, monkeypatch):  # noqa: ARG001
    """Test that .env file parsing handles different formats correctly"""
    # Clear any existing OS environment variables
    monkeypatch.delenv("TEST_VAR", raising=False)
    monkeypatch.delenv("SPACES_VAR", raising=False)
    monkeypatch.delenv("EMPTY_VAR", raising=False)
    monkeypatch.delenv("COMMENT", raising=False)

    with open(mock_env_file / ".env", "w") as f:
        f.write(
            """
                TEST_VAR=value1
                SPACES_VAR  =  value2
                EMPTY_VAR=
                #COMMENT=value3
                INVALID_LINE
                """,
        )

    assert read_env_var("TEST_VAR") == "value1"
    assert read_env_var("SPACES_VAR") == "value2"
    assert read_env_var("EMPTY_VAR") == ""
    assert read_env_var("COMMENT") is None
