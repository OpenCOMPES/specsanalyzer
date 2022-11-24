"""This is the SpecsScan core class

"""
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
from specsanalyzer.metadata import MetaHandler
from specsanalyzer.settings import parse_config

from specsscan.helpers import find_scan
from specsscan.helpers import get_coords
from specsscan.helpers import load_images
from specsscan.helpers import parse_info_to_dict
from specsscan.helpers import parse_lut_to_df

# from specsanalyzer.io import to_h5, load_h5, to_tiff, load_tiff

package_dir = os.path.dirname(find_spec("specsscan").origin)


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
        self._config = parse_config(
            config,
            default_config=f"{package_dir}/config/default.yaml",
        )
        try:
            self.spa = SpecsAnalyzer(config=self._config["spa_params"])
        except KeyError:
            self.spa = SpecsAnalyzer()

    def load_scan(  # pylint:disable=too-many-locals
        self,
        scan: int,
        path: Union[str, Path] = "",
        iterations: Union[
            np.ndarray,
            slice,
            Sequence[int],
            Sequence[slice],
        ] = None,
        delays: Union[
            np.ndarray,
            slice,
            int,
            Sequence[int],
            Sequence[slice],
        ] = None,
    ) -> xr.DataArray:
        """Load scan with given scan number. When iterations is
            given, average is performed over the iterations with an
            option to select the delays via the delays argument.

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
            delays: A 1-D array of the index of delays that the
                data should contain. The array can be a list,
                numpy array or a Tuple consisting of slice objects
                and integers. For ex., np.s_[1:10, 15, -1] would
                be a valid input for delays.
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

        if iterations is None and delays is not None:
            raise ValueError(
                "Invalid input. Delays can only be provided along "
                "with iterations. For selecting delays, slice the resulting "
                "xarray without passing delays. For averaging over the delays "
                "use the check_scan function instead.",
            )

        data = load_images(
            scan_path=path,
            df_lut=df_lut,
            iterations=iterations,
            delays=delays,
        )

        self._scan_info = parse_info_to_dict(path)
        # self._scan_info['name'] = "scan_info_meta"
        # self._attributes.add(self._scan_info)

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

        # res_xarray.attrs["metadata"] = self._attributes

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
        )
        self._scan_info = parse_info_to_dict(path)

        (lens_mode, kin_energy, pass_energy, work_function) = (
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
        res_xarray = xr.concat(
            xr_list,
            dim=xr.DataArray(
                np.arange(0, len(data)),  # slice coords for aborted/ongoing scans
                dims="Iteration",
                name="Iteration",
            ),
        )
        res_xarray = res_xarray.transpose("Angle", "Ekin", "Iteration")

        return res_xarray
