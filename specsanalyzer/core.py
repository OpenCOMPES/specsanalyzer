"""This is the specsanalyzer core class

"""
import os
from typing import Union

from specsanalyzer import io
from specsanalyzer.metadata import MetaHandler
from specsanalyzer.settings import parse_config

# from pathlib import Path
# from typing import Sequence
# import xarray as xr

# from typing import Any
# from typing import List
# from typing import Tuple
# import numpy as np
# from .convert import convert_image

package_dir = os.path.dirname(__file__)


class SpecsAnalyzer:  # pylint: disable=dangerous-default-value
    """[summary]"""

    def __init__(
        self,
        metadata: dict = {},
        config: Union[dict, str] = {},
    ):

        self._config = parse_config(config)

        try:
            self._config["calib2d_dict"] = io.parse_calib2d_to_dict(
                self._config["calib2d_file"],
            )
        except FileNotFoundError:  # default location relative to package directory
            self._config["calib2d_dict"] = io.parse_calib2d_to_dict(
                os.path.join(package_dir, self._config["calib2d_file"]),
            )

        self._attributes = MetaHandler(meta=metadata)

    def __repr__(self):
        if self._config is None:
            pretty_str = "No configuration available"
        else:
            for key in self._config:
                pretty_str += print(f"{self._config[key]}\n")
        # TODO Proper report with scan number, dimensions, configuration etc.
        return pretty_str if pretty_str is not None else ""

    @property
    def config(self):
        """Get config"""
        return self._config

    @config.setter
    def config(self, config: Union[dict, str]):
        """Set config"""
        self._config = parse_config(config)
