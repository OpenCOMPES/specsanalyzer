"""This is the SpecsScan core class"""
from __future__ import annotations

import copy
import os
import pathlib
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from typing import Sequence

import matplotlib
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from specsanalyzer import SpecsAnalyzer
from specsanalyzer.config import parse_config
from specsanalyzer.config import save_config
from specsanalyzer.io import to_h5
from specsanalyzer.io import to_nexus
from specsanalyzer.io import to_tiff
from specsanalyzer.logging import set_verbosity
from specsanalyzer.logging import setup_logging
from specsscan.helpers import get_coords
from specsscan.helpers import get_scan_path
from specsscan.helpers import handle_meta
from specsscan.helpers import load_images
from specsscan.helpers import parse_info_to_dict
from specsscan.helpers import parse_lut_to_df


package_dir = os.path.dirname(find_spec("specsscan").origin)

# Configure logging
logger = setup_logging("specsscan")


class SpecsScan:
    """SpecsAnalyzer class for loading scans and data from SPECS Phoibos electron analyzers,
    generated with the ARPESControl software at Fritz Haber Institute, Berlin, and EPFL, Lausanne.

    Args:
        metadata (dict, optional): Metadata dictionary. Defaults to {}.
        config (Union[dict, str], optional): Metadata dictionary or file path. Defaults to {}.
        **kwds: Keyword arguments passed to ``parse_config``.
    """

    def __init__(
        self,
        metadata: dict = {},
        config: dict | str = {},
        verbose: bool = True,
        **kwds,
    ):
        """SpecsScan constructor.

        Args:
            metadata (dict, optional): Metadata dictionary. Defaults to {}.
            config (Union[dict, str], optional): Metadata dictionary or file path. Defaults to {}.
            verbose (bool, optional): Disable info logs if set to False.
            **kwds: Keyword arguments passed to ``parse_config``.
        """
        self._config = parse_config(
            config,
            default_config=f"{package_dir}/config/default.yaml",
            **kwds,
        )

        set_verbosity(logger, verbose)

        self.metadata = metadata

        self._scan_info: dict[Any, Any] = {}

        try:
            self.spa = SpecsAnalyzer(
                config=self._config["spa_params"],
                folder_config={},
                user_config={},
                system_config={},
                verbose=verbose,
            )
        except KeyError:
            self.spa = SpecsAnalyzer(
                folder_config={},
                user_config={},
                system_config={},
                verbose=verbose,
            )

        self._result: xr.DataArray = None

    # pylint: disable=duplicate-code
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
    def config(self, config: dict | str):
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
        path: str | Path = "",
        iterations: np.ndarray | slice | Sequence[int] | Sequence[slice] = None,
        metadata: dict = {},
        collect_metadata: bool = False,
        **kwds,
    ) -> xr.DataArray:
        """Load scan with given scan number. When iterations is given, average is performed over
        the iterations over all delays.

        Args:
            scan (int): The scan number of interest
            path (str | Path, optional): Either a string of the path to the folder containing the
                scan or a Path object. Defaults to "".
            iterations (np.ndarray | slice | Sequence[int] | Sequence[slice], optional):
                A 1-D array of the number of iterations over which the images are to be averaged.
                The array can be a list, numpy array or a Tuple consisting of slice objects and
                integers. For ex., ``np.s_[1:10, 15, -1]`` would be a valid input for iterations.
                Defaults to None.
            metadata (dict, optional): Metadata dictionary with additional metadata for the scan.
                Defaults to empty dictionary.
            collect_metadata (bool, optional): Option to collect further metadata e.g. from EPICS
                archiver needed for NeXus conversion. Defaults to False.
            **kwds: Additional arguments passed to ``SpecsAnalyzer.convert()``.

        Returns:
            xr.DataArray: xarray DataArray object with kinetic energy, angle/position and
            optionally a third scanned axis (for ex., delay, temperature) as coordinates.
        """
        token = kwds.pop("token", None)
        scan_path = get_scan_path(path, scan, self._config["data_path"])
        df_lut = parse_lut_to_df(scan_path)

        data = load_images(
            scan_path=scan_path,
            df_lut=df_lut,
            iterations=iterations,
            tqdm_enable_nested=self._config["enable_nested_progress_bar"],
        )

        self._scan_info = parse_info_to_dict(scan_path)

        loader_dict = {
            "iterations": iterations,
            "scan_path": str(scan_path),
            "raw_data": data,
        }

        (scan_type, lens_mode, kin_energy, pass_energy, work_function) = (
            self._scan_info["ScanType"],
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
        )

        coords, dim = get_coords(
            scan_path=scan_path,
            scan_type=scan_type,
            scan_info=self._scan_info,
            df_lut=df_lut,
        )

        if scan_type == "single":
            res_xarray = self.spa.convert_image(
                raw_img=data[0],
                lens_mode=lens_mode,
                kinetic_energy=kin_energy,
                pass_energy=pass_energy,
                work_function=work_function,
                **kwds,
            )
        elif scan_type == "voltage":  # and dim == "kinetic energy":
            res_xarray = self.process_sweep_scan(
                raw_data=data,
                kinetic_energy=coords,
                pass_energy=pass_energy,
                lens_mode=lens_mode,
                work_function=work_function,
                **kwds,
            )
        else:
            xr_list: list[xr.DataArray] = []
            for image in data:
                xr_list.append(
                    self.spa.convert_image(
                        raw_img=image,
                        lens_mode=lens_mode,
                        kinetic_energy=kin_energy,
                        pass_energy=pass_energy,
                        work_function=work_function,
                        **kwds,
                    ),
                )
                self.spa.print_msg = False
            self.spa.print_msg = True

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

        slow_axes = {dim} if dim else set()
        fast_axes = set(res_xarray.dims) - slow_axes
        projection = "reciprocal" if "Angle" in fast_axes else "real"
        conversion_metadata = res_xarray.attrs.pop("conversion_parameters")

        # rename coords and store mapping information, if available
        coordinate_mapping = self._config.get("coordinate_mapping", {})
        coordinate_depends = self._config.get("coordinate_depends", {})
        coordinate_labels = self._config.get("coordinate_labels", {})
        rename_dict = {
            k: coordinate_mapping[k] for k in coordinate_mapping.keys() if k in res_xarray.dims
        }
        depends_dict = {
            rename_dict.get(k, k): coordinate_depends[k]
            for k in coordinate_depends.keys()
            if k in res_xarray.dims
        }
        label_dict = {
            rename_dict.get(k, k): coordinate_labels[k]
            for k in coordinate_labels.keys()
            if k in res_xarray.dims
        }

        # store data for resolved axis coordinates
        for axis in res_xarray.dims:
            self._scan_info[axis] = res_xarray.coords[axis].data

        res_xarray = res_xarray.rename(rename_dict)
        for k, v in coordinate_mapping.items():
            if k in fast_axes:
                fast_axes.remove(k)
                fast_axes.add(v)
            if k in slow_axes:
                slow_axes.remove(k)
                slow_axes.add(v)
        self._scan_info["coordinate_depends"] = depends_dict
        self._scan_info["coordinate_label"] = label_dict

        for name in res_xarray.dims:
            try:
                res_xarray[name].attrs["unit"] = self._config["units"][name]
                res_xarray[name].attrs["long_name"] = label_dict[name]
            except KeyError:
                pass

        # reset metadata
        self.metadata = {}

        self.metadata.update(
            **handle_meta(
                df_lut=df_lut,
                scan_info=self._scan_info,
                config=self.config.get("metadata", {}),
                scan=scan,
                fast_axes=list(fast_axes),  # type: ignore
                slow_axes=list(slow_axes),
                projection=projection,
                metadata=copy.deepcopy(metadata),
                collect_metadata=collect_metadata,
                token=token,
            ),
            **{"loader": loader_dict},
            **{"conversion_parameters": conversion_metadata},
        )

        # shift energy axis
        photon_energy = 0.0
        try:
            photon_energy = self.metadata["instrument"]["beam"]["probe"]["incident_energy"]
        except KeyError:
            try:
                photon_energy = self.metadata["elabFTW"]["laser_status"]["probe_photon_energy"]
            except KeyError:
                pass

        self.metadata["scan_info"]["reference_energy"] = "vacuum level"
        if photon_energy and self._config["shift_by_photon_energy"]:
            logger.info(f"Shifting energy axis by photon energy: -{photon_energy} eV")
            res_xarray = res_xarray.assign_coords(
                {
                    rename_dict["Ekin"]: (
                        rename_dict["Ekin"],
                        res_xarray[rename_dict["Ekin"]].data - photon_energy,
                        res_xarray[rename_dict["Ekin"]].attrs,
                    ),
                },
            )
            self.metadata["scan_info"]["reference_energy"] = "Fermi edge"
            self.metadata["scan_info"]["coordinate_label"][rename_dict["Ekin"]] = "E-E_F"
            res_xarray[rename_dict["Ekin"]].attrs["long_name"] = "E-E_F"

        res_xarray.attrs["metadata"] = self.metadata
        self._result = res_xarray

        return res_xarray

    def crop_tool(self, scan: int = None, path: Path | str = "", **kwds):
        """Cropping tool interface to crop_tool method of the SpecsAnalyzer class.

        Args:
            scan (int, optional): Scan number to load data from. Defaults to None.
            path (Path | str, optional): Path in which to search the scan.
                Defaults to config['data_path'].

        Raises:
            ValueError: Raised if no image loaded, and scan not provided
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        if scan is not None:
            scan_path = get_scan_path(path, scan, self._config["data_path"])

            data = load_images(
                scan_path=scan_path,
                tqdm_enable_nested=self._config["enable_nested_progress_bar"],
            )
            image = data[0]
            self._scan_info = parse_info_to_dict(scan_path)
        else:
            try:
                image = self.metadata["loader"]["raw_data"][0]
            except KeyError as exc:
                raise ValueError("No image loaded, load image first!") from exc

        self.spa.crop_tool(
            image,
            self._scan_info["LensMode"],
            self._scan_info["KineticEnergy"],
            self._scan_info["PassEnergy"],
            self._scan_info["WorkFunction"],
            **kwds,
        )

    def save_crop_params(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated crop parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "specs_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "specs_config.yaml"
        if "ek_range_min" not in self.spa.config:
            raise ValueError("No crop parameters to save!")

        config = {
            "spa_params": {
                "crop": self.spa.config["crop"],
                "ek_range_min": self.spa.config["ek_range_min"],
                "ek_range_max": self.spa.config["ek_range_max"],
                "ang_range_min": self.spa.config["ang_range_min"],
                "ang_range_max": self.spa.config["ang_range_max"],
                "angle_offset_px": self.spa.config["angle_offset_px"],
                "rotation_angle": self.spa.config["rotation_angle"],
            },
        }
        save_config(config, filename, overwrite)
        logger.info(f'Saved crop parameters to "{filename}".')

    def fft_tool(self, scan: int = None, path: Path | str = "", **kwds):
        """FFT tool to play around with the peak parameters in the Fourier plane. Built to filter
        out the meshgrid appearing in the raw data images. The optimized parameters are stored in
        the class config dict under fft_filter_peaks.

        Args:
            scan (int, optional): Scan number to load. Defaults to the previously loaded scan.
            path (Path | str): Path from where to load the data. Defaults to config value.
            **kwds: Keyword arguments passed to ``SpecsAnalyzer.fft_tool()``:

                - `apply`: Option to directly apply the settings.
                - `amplitude`: Normalized amplitude of subtraction
                - `pos_x`: horizontal spatial frequency of th mesh
                - `pos_y`: vertical spatial frequency of the mesh
                - `sigma_x`: horizontal frequency width
                - `sigma_y`: vertical frequency width
        """
        matplotlib.use("module://ipympl.backend_nbagg")
        if scan is not None:
            scan_path = get_scan_path(path, scan, self._config["data_path"])

            data = load_images(
                scan_path=scan_path,
                tqdm_enable_nested=self._config["enable_nested_progress_bar"],
            )
            image = data[0]
        else:
            try:
                image = self.metadata["loader"]["raw_data"][0]
            except KeyError as exc:
                raise ValueError("No image loaded, load image first!") from exc

        self.spa.fft_tool(
            image,
            **kwds,
        )

    def save_fft_params(
        self,
        filename: str = None,
        overwrite: bool = False,
    ):
        """Save the generated fft filter parameters to the folder config file.

        Args:
            filename (str, optional): Filename of the config dictionary to save to.
                Defaults to "specs_config.yaml" in the current folder.
            overwrite (bool, optional): Option to overwrite the present dictionary.
                Defaults to False.
        """
        if filename is None:
            filename = "specs_config.yaml"
        if len(self.spa.config["fft_filter_peaks"]) == 0:
            raise ValueError("No fft parameters to save!")

        config = {
            "spa_params": {
                "fft_filter_peaks": self.spa.config["fft_filter_peaks"],
                "apply_fft_filter": self.spa.config["apply_fft_filter"],
            },
        }
        save_config(config, filename, overwrite)
        logger.info(f'Saved fft parameters to "{filename}".')

    def check_scan(
        self,
        scan: int,
        delays: Sequence[int] | int,
        path: str | Path = "",
        metadata: dict = {},
        collect_metadata: bool = False,
        **kwds,
    ) -> xr.DataArray:
        """Function to explore a given 3-D scan as a function of iterations for a given range of
        delays.

        Args:
            scan (int): The scan number of interest
            delays (Sequence[int] | int): A single delay index or a sequence of delay indices
                to be averaged over.
            path (str | Path, optional): Either a string of the path to the folder containing the
                scan or a Path object. Defaults to config['data_path].
            metadata (dict, optional): Metadata dictionary with additional metadata for the scan.
                Defaults to empty dictionary.
            collect_metadata (bool, optional): Option to collect further metadata e.g. from EPICS
                archiver needed for NeXus conversion. Defaults to False.
            **kwds: Keyword arguments passed to ``SpecsAnalyzer.convert()``.

        Raises:
            ValueError: Raised if a single image scan was selected

        Returns:
            xr.DataArray: 3-D xarray of dimensions (Ekin, Angle, Iterations)
        """
        token = kwds.pop("token", None)
        scan_path = get_scan_path(path, scan, self._config["data_path"])
        df_lut = parse_lut_to_df(scan_path)

        data = load_images(
            scan_path=scan_path,
            df_lut=df_lut,
            delays=delays,
            tqdm_enable_nested=self._config["enable_nested_progress_bar"],
        )

        self._scan_info = parse_info_to_dict(scan_path)

        loader_dict = {
            "delays": delays,
            "scan_path": str(scan_path),
            "raw_data": data,
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

        conversion_metadata = xr_list[0].attrs["conversion_parameters"]

        dims = get_coords(  # noqa: F841
            scan_path=scan_path,
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
                res_xarray[name].attrs["unit"] = self._config["units"][name]
            except KeyError:
                pass

        slow_axes = {"Iteration"}
        fast_axes = set(res_xarray.dims) - slow_axes
        projection = "reciprocal" if "Angle" in fast_axes else "real"

        # reset metadata
        self.metadata = {}

        self.metadata.update(
            **handle_meta(
                df_lut=df_lut,
                scan_info=self._scan_info,
                config=self.config.get("metadata", {}),
                scan=scan,
                fast_axes=list(fast_axes),  # type: ignore
                slow_axes=list(slow_axes),
                projection=projection,
                metadata=copy.deepcopy(metadata),
                collect_metadata=collect_metadata,
                token=token,
            ),
            **{"loader": loader_dict},
            **{"conversion_parameters": conversion_metadata},
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

            **kwds: Keyword arguments, which are passed to the writer functions:
                For TIFF writing:

                - **alias_dict**: Dictionary of dimension aliases to use.

                For HDF5 writing:

                - **mode**: hdf5 read/write mode. Defaults to "w".

                For NeXus:

                - **reader**: Name of the pynxtools reader to use.
                  Defaults to config["nexus"]["reader"]
                - **definition**: NeXus application definition to use for saving.
                  Must be supported by the used ``reader``. Defaults to
                  config["nexus"]["definition"]
                - **input_files**: A list of input files to pass to the reader.
                  Defaults to config["nexus"]["input_files"]
                - **eln_data**: Path to a json file with data from an electronic lab notebook.
                  Its is appended to the ``input_files``.
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
                copy.deepcopy(self._config["nexus"]["input_files"]),
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

    def process_sweep_scan(
        self,
        raw_data: list[np.ndarray],
        kinetic_energy: np.ndarray,
        pass_energy: float,
        lens_mode: str,
        work_function: float,
        **kwds,
    ) -> xr.DataArray:
        """Process sweep scan by interpolating each frame onto a common grid given by the voltage
        step, and summing over all frames.

        Args:
            raw_data (list[np.ndarray]): List of raw data images
            kinetic_energy (np.ndarray): Array of analyzer set kinetic energy values
            pass_energy (float): set analyzer pass energy
            lens_mode (str): analyzer lens mode, check calib2d for a list of modes CamelCase naming
                convention e.g. "WideAngleMode"
            work_function (float): set analyzer work function

        Returns:
            xr.DataArray: Converted sweep scan
        """
        ekin_step = kinetic_energy[1] - kinetic_energy[0]
        if not (np.diff(kinetic_energy) == ekin_step).all():
            logger.warning(
                "Conversion of sweep scans with non-equidistant energy steps "
                "might produce wrong results!",
            )

        # convert first image
        converted = self.spa.convert_image(
            raw_data[0],
            lens_mode,
            kinetic_energy[0],
            pass_energy,
            work_function,
            **kwds,
        )
        conversion_parameters = converted.attrs["conversion_parameters"]
        # check for crop parameters
        if (
            not {"ang_range_min", "ang_range_max", "ek_range_min", "ek_range_max"}.issubset(
                set(self.spa.config.keys()),
            )
            or not self.spa.config["crop"]
        ):
            logger.warning("No valid cropping parameters found, consider using crop_tool() to set.")

        e_step = converted.Ekin[1] - converted.Ekin[0]
        e0 = converted.Ekin[-1] - ekin_step
        e1 = converted.Ekin[0] + kinetic_energy[-1] - kinetic_energy[0]
        data = xr.DataArray(
            data=np.zeros((len(converted.Angle), len(np.arange(e0, e1, e_step)))),
            coords={"Angle": converted.Angle, "Ekin": np.arange(e0, e1, e_step)},
            dims=["Angle", "Ekin"],
        )
        for i, ekin in enumerate(tqdm(kinetic_energy)):
            self.spa.print_msg = False
            converted = self.spa.convert_image(
                raw_img=raw_data[i],
                lens_mode=lens_mode,
                kinetic_energy=ekin,
                pass_energy=pass_energy,
                work_function=work_function,
                **kwds,
            )
            energies = converted.Ekin.where(
                (converted.Ekin >= data.Ekin[0]) & (converted.Ekin < data.Ekin[-1]),
                0,
            )
            energy_indices = np.argwhere(energies.values).squeeze()
            # filling target array using "nearest" method
            target_energy = data.Ekin.sel(Ekin=converted.Ekin[energies > 0], method="nearest")
            target_indices = np.argwhere(np.isin(data.Ekin.values, target_energy.values)).squeeze()
            data[:, target_indices] += converted[:, energy_indices].values

        self.spa.print_msg = True
        # Strip first and last energy points, as they are not fully filled
        data = data[:, 1:-1]
        data.attrs["conversion_parameters"] = conversion_parameters

        return data
