import json
from pathlib import Path
from typing import Union

import yaml


def parse_config(config: Union[dict, Path, str] = {}) -> dict:
    """Handle config dictionary or files.

    Args:
        ...

    Raises:
        ...

    Returns:
        ...
    """

    if isinstance(config, dict):
        return config

    if isinstance(config, str):
        config_file = Path(config)
    else:
        config_file = Path(config)

    if not isinstance(config_file, Path):
        raise TypeError(
            "config must be either a Path to a config file, or a config dictionary!",
        )

    if config_file.suffix == ".json":
        config_dict = json.load(config_file)
    elif config_file.suffix == ".yaml":
        with open(config_file) as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        raise TypeError("config file must be of type json or yaml!")

    return config_dict
