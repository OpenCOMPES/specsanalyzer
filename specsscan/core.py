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

from specsscan.helpers import find_scan
from specsscan.helpers import get_coords
from specsscan.helpers import load_images
from specsscan.helpers import parse_info_to_dict
from specsscan.helpers import parse_lut_to_df
from specsscan.metadata import MetaHandler
from specsscan.settings import parse_config


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

    def load_scan(  # pylint:disable=too-many-locals
        self,
        scan: int,
        path: Union[str, Path] = "",
        iterations: Union[
            List[int],
            np.ndarray,
            slice,
            Sequence,
        ] = None,  # type: ignore
    ) -> xr.DataArray:
        """Load scan with given scan number.

        Args:
            scan: The run number of interest
            path: Either a string of the path to the folder
                containing the scan or a Path object
            iterations: A 1-D array of the number of iterations over
                which the images are to be averaged. The array
                can be a list, numpy array or a Tuple consisting of
                slice objects and integers. For ex.,
                np.s_[1:10, 15, -1] would be a valid input for
                iterations.

        Raises:
            FileNotFoundError, IndexError

        Returns:
            xres: xarray DataArray object with kinetic energy, angle/position
                and optionally a third scanned axis (for ex., delay, temperature)
                as coordinates.
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

        df_lut = parse_lut_to_df(path)  # TODO: storing metadata from df_lut

        data = load_images(
            scan_path=path,
            df_lut=df_lut,
            iterations=iterations,
        )

        self._scan_info = parse_info_to_dict(path)

        (scan_type, lens_mode, kin_energy, pass_energy, work_function) = (
            self._scan_info["ScanType"],
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
        )

        xr_list = []
        for image in data:
            xr_list.append(
                self.spa.convert_image(
                    image,
                    lens_mode,
                    kin_energy,
                    pass_energy,
                    work_function,
                ),
            )

        coords, dim = get_coords(
            scan_path=path,
            scan_type=scan_type,
            scan_info=self._scan_info,
            df_lut=df_lut,
        )

        if scan_type == "single":
            res_xarray = xr_list[0]
        else:
            res_xarray = xr.concat(
                xr_list,
                dim=xr.DataArray(
                    coords[:len(data)],  # slice coords for aborted/ongoing scans
                    dims=dim,
                    name=dim,
                ),
            )
            if dim in ["polar", "tilt", "azimuth"]:
                res_xarray = res_xarray.transpose("Angle", dim, "Ekin")
            else:
                res_xarray = res_xarray.transpose("Angle", "Ekin", dim)

        return res_xarray  # type:ignore
