from pathlib import Path
from typing import Sequence
from typing import Union

import psutil
import xarray as xr
from specsanalyzer import SpecsAnalyzer

from .metadata import MetaHandler

# from typing import Any
# from typing import List
# from typing import Tuple
# import numpy as np
# from .convert import convert_image

N_CPU = psutil.cpu_count()


class SpecsScan:
    """[summary]"""

    def __init__(
        self,
        metadata: dict = {},
        config: Union[dict, Path, str] = {},
    ):

        # TODO: handle/load config dict/file
        self._config = config
        if not isinstance(self._config, dict):
            self._config = {}
        # Define defaults. TODO
        # if "hist_mode" not in self._config.keys():
        #    self._config["hist_mode"] = "numba"

        self._attributes = MetaHandler(meta=metadata)

        self.spa = SpecsAnalyzer()

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
