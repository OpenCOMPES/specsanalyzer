"""This is the SpecsScan core class

"""
import copy
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
import xarray as xr
from specsanalyzer import SpecsAnalyzer
from specsanalyzer.settings import parse_config

from specsscan.helpers import find_scan
from specsscan.helpers import get_coords
from specsscan.helpers import handle_meta
from specsscan.helpers import load_images
from specsscan.helpers import parse_info_to_dict
from specsscan.helpers import parse_lut_to_df

# from specsanalyzer.io import to_h5, load_h5, to_tiff, load_tiff

package_dir = os.path.dirname(find_spec("specsscan").origin)

default_units = {
    "Angle": "degrees",
    "Ekin": "eV",
    "delay": "fs",
    "mirrorX": "steps",
    "mirrorY": "steps",
    "X": "mm",
    "Y": "mm",
    "Z": "mm",
    "polar": "degrees",
    "tilt": "degrees",
    "azimuth": "degrees",
}


class SpecsScan:
    """[summary]"""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        metadata: dict = {},
        config: Union[dict, str] = {},
    ):

        self._config = parse_config(
            config,
            default_config=f"{package_dir}/config/default.yaml",
        )

        # self.metadata = MetaHandler(meta=metadata)
        self.metadata = metadata

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
        self._config = parse_config(
            config,
            default_config=f"{package_dir}/config/default.yaml",
        )
        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def load_scan(
        self,
        scan: int,
        path: Union[str, Path] = "",
        iterations: Union[
            np.ndarray,
            slice,
            Sequence[int],
            Sequence[slice],
        ] = None,
    ) -> xr.DataArray:
        """Load scan with given scan number. When iterations is
            given, average is performed over the iterations over
            all delays.

        Args:
            scan: The scan number of interest
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
            path = Path(path).joinpath(str(scan).zfill(4))
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
            tqdm_enable_nested=self._config["enable_nested_progress_bar"],
        )

        self._scan_info = parse_info_to_dict(path)
        config_meta = copy.deepcopy(self.config)
        config_meta['spa_params'].pop('calib2d_dict')

        loader_dict = {
            "iterations": iterations,
            "scan_path": path,
            "raw_data": data,
            "convert_config": config_meta,
        }
        self.metadata.update(
            **handle_meta(
                df_lut,
                self._scan_info,
                config_meta,
            ),
            **{"loader": loader_dict},
        )
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
                    coords[
                        : len(data)
                    ],  # slice coords for aborted/ongoing scans
                    dims=dim,
                    name=dim,
                ),
            )
            if dim in ["polar", "tilt", "azimuth"]:
                res_xarray = res_xarray.transpose("Angle", dim, "Ekin")
            else:
                res_xarray = res_xarray.transpose("Angle", "Ekin", dim)

        for name in res_xarray.dims:
            res_xarray[name].attrs['unit'] = default_units[name]

        res_xarray.attrs["metadata"] = self.metadata

        return res_xarray

    def check_scan(
        self,
        scan: int,
        delays: Union[
            Sequence[int],
            int,
        ],
        path: Union[str, Path] = "",
    ) -> xr.DataArray:
        """Function to explore a given 3-D scan as a function
            of iterations for a given range of delays
        Args:
            scan: The scan number of interest
            delay: A single delay index or a range of delay indices
                to be averaged over.
            path: Either a string of the path to the folder
                containing the scan or a Path object
        Raises:
            FileNotFoundError
        Returns:
            A 3-D numpy array of dimensions (Ekin, K, Iterations)
        """

        if path:
            path = Path(path).joinpath(str(scan).zfill(4))
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

        df_lut = parse_lut_to_df(path)

        data = load_images(
            scan_path=path,
            df_lut=df_lut,
            delays=delays,
            tqdm_enable_nested=self._config["enable_nested_progress_bar"],
        )
        self._scan_info = parse_info_to_dict(path)
        config_meta = copy.deepcopy(self.config)
        config_meta['spa_params'].pop('calib2d_dict')

        loader_dict = {
            "delays": delays,
            "scan_path": path,
            "raw_data": load_images(  # AVG data
                path,
                df_lut,
            ),
            "convert_config": config_meta,
            "check_scan": True,
        }
        self.metadata.update(
            **handle_meta(
                df_lut,
                self._scan_info,
                config_meta,
            ),
            **{"loader": loader_dict},
        )

        (scan_type, lens_mode, kin_energy, pass_energy, work_function) = (
            self._scan_info["ScanType"],
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
        )
        if scan_type == "single":
            raise ValueError(
                "Invalid input. A 3-D scan is expected, "
                "a 2-D single scan was provided instead.",
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
        res_xarray = xr.concat(
            xr_list,
            dim=xr.DataArray(
                np.arange(0, len(data)),  # slice coords for aborted/ongoing scans
                dims="Iteration",
                name="Iteration",
            ),
        )
        res_xarray = res_xarray.transpose("Angle", "Ekin", "Iteration")
        for name in res_xarray.dims:
            try:
                res_xarray[name].attrs['unit'] = default_units[name]
            except KeyError:
                pass

        res_xarray.attrs["metadata"] = self.metadata

        return res_xarray
