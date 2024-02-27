"""This is the SpecsScan core class

"""
import copy
import os
import pathlib
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import matplotlib
import numpy as np
import xarray as xr

from specsanalyzer import SpecsAnalyzer
from specsanalyzer.config import parse_config
from specsanalyzer.io import to_h5
from specsanalyzer.io import to_nexus
from specsanalyzer.io import to_tiff
from specsscan.helpers import find_scan
from specsscan.helpers import get_coords
from specsscan.helpers import handle_meta
from specsscan.helpers import load_images
from specsscan.helpers import parse_info_to_dict
from specsscan.helpers import parse_lut_to_df


package_dir = os.path.dirname(find_spec("specsscan").origin)

default_units: Dict[Any, Any] = {
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
    "voltage": "V",
    "temperature": "K",
}


class SpecsScan:
    """[summary]"""

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        metadata: dict = {},
        config: Union[dict, str] = {},
        **kwds,
    ):

        self._config = parse_config(
            config,
            default_config=f"{package_dir}/config/default.yaml",
            **kwds,
        )

        # self.metadata = MetaHandler(meta=metadata)
        self.metadata = metadata

        self._scan_info: Dict[Any, Any] = {}

        try:
            self.spa = SpecsAnalyzer(
                config=self._config["spa_params"],
                folder_config={},
                user_config={},
                system_config={},
            )
        except KeyError:
            self.spa = SpecsAnalyzer()

        self._result: xr.DataArray = None

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

    @property
    def result(self):
        """Get result xarray"""
        return self._result

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
        metadata: dict = None,
        **kwds,
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
            metadata (dict, optional): Metadata dictionary with additional metadata for the scan
            **kwds: Additional arguments for the SpecsAnalyzer converter. For ex., passing
                crop=True crops the data if cropping data is already present in the given instance.
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
        config_meta["spa_params"].pop("calib2d_dict")

        loader_dict = {
            "iterations": iterations,
            "scan_path": path,
            "raw_data": data,
            "convert_config": config_meta["spa_params"],
        }

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
                    **kwds,
                ),
            )
            self.spa.print_msg = False
        self.spa.print_msg = True

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
                    coords[: len(data)],  # slice coords for aborted/ongoing scans
                    dims=dim,
                    name=dim,
                ),
            )
            if dim in ["polar", "tilt", "azimuth"]:
                res_xarray = res_xarray.transpose("Angle", dim, "Ekin")
            else:
                res_xarray = res_xarray.transpose("Angle", "Ekin", dim)

        for name in res_xarray.dims:
            res_xarray[name].attrs["unit"] = default_units[name]
        self.metadata.update(
            **handle_meta(
                df_lut,
                self._scan_info,
                self.config,
                dim,
            ),
            **{"loader": loader_dict},
        )
        if metadata is not None:
            self.metadata.update(**metadata)

        res_xarray.attrs["metadata"] = self.metadata
        self._result = res_xarray

        return res_xarray

    def crop_tool(self):
        """
        Croping tool interface to crop_tool method
        of the SpecsAnalyzer class.
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        image = self.metadata["loader"]["raw_data"][0]
        self.spa.crop_tool(
            image,
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
        )

    def check_scan(
        self,
        scan: int,
        delays: Union[Sequence[int], int],
        path: Union[str, Path] = "",
        metadata: dict = None,
        **kwds,
    ) -> xr.DataArray:
        """Function to explore a given 3-D scan as a function
            of iterations for a given range of delays
        Args:
            scan: The scan number of interest
            delay: A single delay index or a range of delay indices
                to be averaged over.
            path: Either a string of the path to the folder
                containing the scan or a Path object
            metadata (dict, optional): Metadata dictionary with additional metadata for the scan
            **kwds: Additional arguments for the SpecsAnalyzer converter. For ex., passing
                crop=True crops the data if cropping data is already present in the given instance.
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
        config_meta["spa_params"].pop("calib2d_dict")

        loader_dict = {
            "delays": delays,
            "scan_path": path,
            "raw_data": load_images(  # AVG data
                path,
                df_lut,
            ),
            "convert_config": config_meta["spa_params"],
            "check_scan": True,
        }

        (scan_type, lens_mode, kin_energy, pass_energy, work_function) = (
            self._scan_info["ScanType"],
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
        )
        if scan_type == "single":
            raise ValueError(
                "Invalid input. A 3-D scan is expected, a 2-D single scan was provided instead.",
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
                    **kwds,
                ),
            )
            self.spa.print_msg = False
        self.spa.print_msg = True

        dims = get_coords(
            scan_path=path,
            scan_type=scan_type,
            scan_info=self._scan_info,
            df_lut=df_lut,
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
                res_xarray[name].attrs["unit"] = default_units[name]
            except KeyError:
                pass

        self.metadata.update(
            **handle_meta(
                df_lut,
                self._scan_info,
                self.config,
                dims[1],
            ),
            **{"loader": loader_dict},
        )
        if metadata is not None:
            self.metadata.update(**metadata)

        res_xarray.attrs["metadata"] = self.metadata

        self._result = res_xarray

        return res_xarray

    def save(
        self,
        faddr: str,
        **kwds,
    ):
        """Saves the loaded data to the provided path and filename.

        Args:
            faddr (str): Path and name of the file to write. Its extension determines
                the file type to write. Valid file types are:

                - "*.tiff", "*.tif": Saves a TIFF stack.
                - "*.h5", "*.hdf5": Saves an HDF5 file.
                - "*.nxs", "*.nexus": Saves a NeXus file.

            **kwds: Keyword argumens, which are passed to the writer functions:
                For TIFF writing:

                - **alias_dict**: Dictionary of dimension aliases to use.

                For HDF5 writing:

                - **mode**: hdf5 read/write mode. Defaults to "w".

                For NeXus:

                - **reader**: Name of the nexustools reader to use.
                  Defaults to config["nexus"]["reader"]
                - **definiton**: NeXus application definition to use for saving.
                  Must be supported by the used ``reader``. Defaults to
                  config["nexus"]["definition"]
                - **input_files**: A list of input files to pass to the reader.
                  Defaults to config["nexus"]["input_files"]
        """
        if self._result is None:
            raise NameError("Need to load data first!")

        extension = pathlib.Path(faddr).suffix

        if extension in (".tif", ".tiff"):
            to_tiff(
                data=self._result,
                faddr=faddr,
                **kwds,
            )
        elif extension in (".h5", ".hdf5"):
            to_h5(
                data=self._result,
                faddr=faddr,
                **kwds,
            )
        elif extension in (".nxs", ".nexus"):
            reader = kwds.pop("reader", self._config["nexus"]["reader"])
            definition = kwds.pop(
                "definition",
                self._config["nexus"]["definition"],
            )
            input_files = kwds.pop(
                "input_files",
                self._config["nexus"]["input_files"],
            )
            if isinstance(input_files, str):
                input_files = [input_files]

            if "eln_data" in kwds:
                input_files.append(kwds.pop("eln_data"))

            to_nexus(
                data=self._result,
                faddr=faddr,
                reader=reader,
                definition=definition,
                input_files=input_files,
                **kwds,
            )

        else:
            raise NotImplementedError(
                f"Unrecognized file format: {extension}.",
            )
