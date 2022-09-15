"""This is the SpecsScan core class

"""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

import numpy as np
import xarray as xr
from specsanalyzer import SpecsAnalyzer

from specsscan.metadata import MetaHandler
from specsscan.settings import parse_config

# from typing import Any
# from typing import List
# from typing import Tuple
# import numpy as np
# from .convert import convert_image


class SpecsScan:
    """[summary]"""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        metadata: dict = {},
        config: Union[dict, str] = {},
    ):

        self._config = parse_config(config)

        self._attributes = MetaHandler(meta=metadata)

        self._scan_info: Dict[Any, Any] = {}

        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def __repr__(self):
        if self._config is None:
            pretty_str = "No configuration available"
        else:
            pretty_str = ""
            for k in self._config:
                pretty_str += f"{k} = {self._config[k]}\n"
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
        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def load_scan(
        self,
        scan: int,
        path: Union[str, Path] = "",
        cycles: Sequence = None,
        **kwds,
    ) -> xr.DataArray:
        """Load scan with given scan number.

        Args:
            scan: The run number of interest
            path: Either a string of the path to the folder
                containing the scan or a Path object
            cycles:

        Raises:
            FileNotFoundError

        Returns:
            xres: xarray DataArray object with kinetic energy and angle/position as
                coordinates
        """
        if path:
            path = Path(path).joinpath(str(scan))
            if not path.is_dir():
                raise FileNotFoundError(
                    f"The provided path {path} was not found.",
                )
        else:
            # search for the given scan using the default path
            path = Path(self._config["data_path"])
            # path_scan = sorted(path.glob(f"20[1,2][9,0-9]/*/*/Raw Data/{scan}"))
            path_scan_list = find_scan(path, scan)
            if not path_scan_list:
                raise FileNotFoundError(
                    f"Scan number {scan} not found",
                )
            path = path_scan_list[0]

        try:
            self._scan_info = parse_info_to_dict(path)

        except FileNotFoundError:
            print("info.txt file not found.")
            # raise FileNotFoundError("info.txt file not found.")
            raise
        (scan_type, lens_mode, kin_energy, pass_energy) = (
            self._scan_info["ScanType"],
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
        )

        # Treat the data based on the scan type.

        return xr.DataArray(np.zeros((10, 10), float))


def parse_info_to_dict(path: Path) -> Dict:
    """Parses the contents of info.txt file
        into a dictionary
    Args:
        path: Path object pointing to the scan folder
    Returns:
        info_dict: Parsed dictionary
    """
    info_dict: Dict[Any, Any] = {}
    with open(path.joinpath("info.txt"), encoding="utf-8") as info_file:

        for line in info_file.readlines():

            if "=" in line:  # older scans
                line_list = line.rstrip("\nV").split("=")

            elif ":" in line:
                line_list = line.rstrip("\nV").split(":")

            else:
                continue

            k, v = line_list[0], line_list[1]

            try:
                info_dict[k] = float(v)
            except ValueError:
                info_dict[k] = v

    return info_dict


def find_scan(path: Path, scan: int) -> List[Path]:
    """Search function to locate the scan folder
    Args:
        path: Path object for data from the default config file
        scan: Scan number of the scan of interest
    Returns:
        scan_path: Path object pointing to the scan folder
    """
    print("Scan path not provided, searching directories...")
    for file in path.iterdir():

        if file.is_dir():

            try:
                base = int(file.stem)

            except ValueError:  # not numeric
                continue

            if base >= 2019:  # only look at folders 2019 onwards

                scan_path = sorted(
                    file.glob(f"*/*/Raw Data/{scan}"),
                )
                if scan_path:
                    print("Scan found at path:", scan_path[0])
                    break
    return scan_path
