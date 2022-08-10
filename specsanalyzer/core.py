import os
from pathlib import Path
from typing import Sequence
from typing import Union

import xarray as xr

import specsanalyzer
from specsanalyzer import io
from specsanalyzer.metadata import MetaHandler
from specsanalyzer.settings import parse_config

# from typing import Any
# from typing import List
# from typing import Tuple
# import numpy as np
# from .convert import convert_image

package_dir = os.path.dirname(specsanalyzer.__file__)


class SpecsAnalyzer:
    """[summary]"""

    def __init__(
        self,
        metadata: dict = {},
        config: Union[dict, Path, str] = {},
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
            s = "No configuration available"
        else:
            s = print(self._config)
        # TODO Proper report with scan number, dimensions, configuration etc.
        return s if s is not None else ""

    def load_scan(
        self,
        scan: int,
        path: Union[str, Path] = "",
        cycles: Sequence = None,
        **kwds,
    ) -> xr.DataArray:
        """Load scan with given scan number.

        Args:
            ...

        Raises:
            ...

        Returns:
            ...
        """

    pass
