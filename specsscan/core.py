from pathlib import Path
from typing import Dict
from typing import Sequence
from typing import Union

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

    def __init__(
        self,
        metadata: dict = {},
        config: Union[dict, str] = {},
    ):

        self._config = parse_config(config)

        self._attributes = MetaHandler(meta=metadata)

        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def __repr__(self):
        if self._config is None:
            s = "No configuration available"
        else:
            s = print(self._config)
        # TODO Proper report with scan number, dimensions, configuration etc.
        return s if s is not None else ""

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Union[dict, str]):
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
            if path.is_dir():
                try:
                    self._config["info_params"] = parse_info_to_dict(path)
                except FileNotFoundError:
                    # Ask params in kwargs instead?
                    raise FileNotFoundError("info.txt file not found.")
            else:
                raise FileNotFoundError(
                    f"The provided path {path} was not found.",
                )
        else:
            # search for the given scan using the default path
            path = Path(self._config['data_path'])
            # takes upto 55 seconds to look for the folder using glob
            # path_scan = sorted(path.glob(f"20[1,2][9,0-9]/*/*/Raw Data/{scan}"))[0]
            # takes upto 23 seconds using user defined function for the same scan
            path_scan = find_scan(path, scan)
            if path_scan:
                try:
                    self._config["info_params"] = parse_info_to_dict(path_scan[0])
                except FileNotFoundError:
                    raise FileNotFoundError("info.txt file not found")
            else:
                raise FileNotFoundError(
                    f"Scan number {scan} not found",
                )

        (
            scan_type,
            lens_mode,
            kin_energy,
            pass_energy,
        ) = (
            self._config["info_params"]["ScanType"],
            self._config["info_params"]["LensMode"],
            self._config["info_params"]["KineticEnergy"],
            self._config["info_params"]["PassEnergy"],
        )

        # Treat the data based on the scan type.


def parse_info_to_dict(path: Path) -> Dict:
    """Parses the contents of info.txt file
        into a dictionary
    Args:
        path: Path object pointing to the scan folder
    Returns:
        info_dict: Parsed dictionary
    """
    info_dict = {}
    with open(path.joinpath("info.txt")) as info_file:
        for line in info_file.readlines():
            if line.startswith("\n"):
                continue
            if "=" in line:  # older scans
                line_list = line.rstrip("\nV").split("=")
            elif ":" in line:
                line_list = line.rstrip("\nV").split(":")
            k, v = line_list[0], line_list[1]
            try:
                v = float(v)
            except ValueError:
                pass
            info_dict[k] = v

    return info_dict


def find_scan(path: Path, scan: int):
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
            if base >= 2019:  # only look at folders after 2019
                print(file)
                scan_path = sorted(
                    file.glob(f"*/*/Raw Data/{scan}"),
                )
                if scan_path:
                    break
    return scan_path
