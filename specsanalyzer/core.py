"""This is the specsanalyzer core class"""
from __future__ import annotations

import os
from typing import Any
from typing import Generator

import imutils
import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display

from specsanalyzer import io
from specsanalyzer.config import complete_dictionary
from specsanalyzer.config import parse_config
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer.convert import get_damatrix_fromcalib2d
from specsanalyzer.convert import physical_unit_data
from specsanalyzer.img_tools import crop_xarray
from specsanalyzer.img_tools import fourier_filter_2d

package_dir = os.path.dirname(__file__)


class SpecsAnalyzer:
    """SpecsAnalyzer: A class to convert photoemission data from a SPECS Phoibos analyzer from
    camera image coordinates into physical units (energy, angle, position).

    Args:
        metadata (dict, optional): Metadata dictionary. Defaults to {}.
        config (dict  | str, optional): Metadata dictionary or file path. Defaults to {}.
        **kwds: Keyword arguments passed to ``parse_config``.
    """

    def __init__(
        self,
        metadata: dict[Any, Any] = {},
        config: dict[Any, Any] | str = {},
        **kwds,
    ):
        """SpecsAnalyzer constructor.

        Args:
            metadata (dict, optional): Metadata dictionary. Defaults to {}.
            config (dict | str, optional): Metadata dictionary or file path. Defaults to {}.
            **kwds: Keyword arguments passed to ``parse_config``.
        """
        self._config = parse_config(
            config,
            **kwds,
        )
        self.metadata = metadata
        self._data_array = None
        self.print_msg = True
        try:
            self._calib2d = io.parse_calib2d_to_dict(self._config["calib2d_file"])

        except FileNotFoundError:  # default location relative to package directory
            self._calib2d = io.parse_calib2d_to_dict(
                os.path.join(package_dir, self._config["calib2d_file"]),
            )

        self._correction_matrix_dict: dict[Any, Any] = {}

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
    def config(self) -> dict:
        """Get config"""
        return self._config

    @property
    def calib2d(self) -> dict:
        """Get calib2d dict"""
        return self._calib2d

    @property
    def correction_matrix_dict(self) -> dict:
        """Get correction_matrix_dict"""
        return self._correction_matrix_dict

    def convert_image(
        self,
        raw_img: np.ndarray,
        lens_mode: str,
        kinetic_energy: float,
        pass_energy: float,
        work_function: float,
        conversion_parameters: dict = None,
        **kwds,
    ) -> xr.DataArray:
        """Converts an imagin in physical unit data, angle vs energy

        Args:
            raw_img (np.ndarray): Raw image data, numpy 2d matrix
            lens_mode (str): analzser lens mode, check calib2d for a list of modes CamelCase naming
                convention e.g. "WideAngleMode"
            kinetic_energy (float): set analyser kinetic energy
            pass_energy (float): set analyser pass energy
            work_function (float): set analyser work function
            conversion_parameters (dict, optional): dictionary of conversion parameters,
                overwriting determination from calib2d file. Defaults to None.

        Returns:
            xr.DataArray: xarray containg the corrected data and kinetic and angle axis
        """
        if conversion_parameters is None:
            conversion_parameters = {}
        else:
            conversion_parameters = conversion_parameters.copy()

        if "apply_fft_filter" not in conversion_parameters.keys():
            conversion_parameters["apply_fft_filter"] = kwds.pop(
                "apply_fft_filter",
                self._config.get("apply_fft_filter", False),
            )
        if "binning" not in conversion_parameters.keys():
            conversion_parameters["binning"] = kwds.pop("binning", self._config.get("binning", 1))
        if "rotation_angle" not in conversion_parameters.keys():
            conversion_parameters["rotation_angle"] = kwds.pop(
                "rotation_angle",
                self._config.get("rotation_angle", 0),
            )

        if conversion_parameters["apply_fft_filter"]:
            try:
                if "fft_filter_peaks" not in conversion_parameters.keys():
                    conversion_parameters["fft_filter_peaks"] = kwds.pop(
                        "fft_filter_peaks",
                        self._config["fft_filter_peaks"],
                    )
                img = fourier_filter_2d(raw_img, conversion_parameters["fft_filter_peaks"])
            except KeyError:
                img = raw_img
                conversion_parameters["apply_fft_filter"] = False
        else:
            img = raw_img

        if conversion_parameters["rotation_angle"]:
            img_rotated = imutils.rotate(img, angle=conversion_parameters["rotation_angle"])
            img = img_rotated

        if "lens_mode" not in conversion_parameters.keys():
            conversion_parameters["lens_mode"] = lens_mode
            conversion_parameters["kinetic_energy"] = kinetic_energy
            conversion_parameters["pass_energy"] = pass_energy
            conversion_parameters["work_function"] = work_function
            # Determine conversion parameters from calib2d
            (
                conversion_parameters["a_inner"],
                conversion_parameters["da_matrix"],
                conversion_parameters["retardation_ratio"],
                conversion_parameters["source"],
                conversion_parameters["dims"],
            ) = get_damatrix_fromcalib2d(
                lens_mode=lens_mode,
                kinetic_energy=kinetic_energy,
                pass_energy=pass_energy,
                work_function=work_function,
                calib2d_dict=self._calib2d,
            )
            conversion_parameters["e_shift"] = np.array(self._calib2d["eShift"])
            conversion_parameters["de1"] = [self._calib2d["De1"]]
            conversion_parameters["e_range"] = self._calib2d["eRange"]
            conversion_parameters["a_range"] = self._calib2d[lens_mode]["default"]["aRange"]
            conversion_parameters["pixel_size"] = (
                self._config["pixel_size"] * self._config["binning"]
            )
            conversion_parameters["magnification"] = self._config["magnification"]
            conversion_parameters["angle_offset_px"] = kwds.get(
                "angle_offset_px",
                self._config.get("angle_offset_px", 0),
            )
            conversion_parameters["energy_offset_px"] = kwds.get(
                "energy_offset_px",
                self._config.get("energy_offset_px", 0),
            )

        # do we need to calculate a new conversion matrix? Check correction matrix dict:
        new_matrix = False
        try:
            old_db = self._correction_matrix_dict[lens_mode][kinetic_energy][pass_energy][
                work_function
            ]

            ek_axis = old_db["ek_axis"]
            angle_axis = old_db["angle_axis"]
            angular_correction_matrix = old_db["angular_correction_matrix"]
            e_correction = old_db["e_correction"]
            jacobian_determinant = old_db["jacobian_determinant"]

        except KeyError:
            new_matrix = True

        if new_matrix or "angle_offset_px" in kwds or "energy_offset_px" in kwds:
            old_matrix_check = False
            (
                ek_axis,
                angle_axis,
                angular_correction_matrix,
                e_correction,
                jacobian_determinant,
            ) = calculate_matrix_correction(
                kinetic_energy=kinetic_energy,
                pass_energy=pass_energy,
                nx_pixels=img.shape[1],
                ny_pixels=img.shape[0],
                pixel_size=conversion_parameters["pixel_size"],
                magnification=conversion_parameters["magnification"],
                e_shift=conversion_parameters["e_shift"],
                de1=conversion_parameters["de1"],
                e_range=conversion_parameters["e_range"],
                a_range=conversion_parameters["a_range"],
                a_inner=conversion_parameters["a_inner"],
                da_matrix=conversion_parameters["da_matrix"],
                angle_offset_px=conversion_parameters["angle_offset_px"],
                energy_offset_px=conversion_parameters["energy_offset_px"],
            )

            # save the config parameters for later use collect the info in a new nested dictionary
            current_correction = {
                lens_mode: {
                    kinetic_energy: {
                        pass_energy: {
                            work_function: {
                                "ek_axis": ek_axis,
                                "angle_axis": angle_axis,
                                "angular_correction_matrix": angular_correction_matrix,
                                "e_correction": e_correction,
                                "jacobian_determinant": jacobian_determinant,
                            },
                        },
                    },
                },
            }

            # add the new lens mode to the correction matrix dict
            self._correction_matrix_dict = complete_dictionary(
                self._correction_matrix_dict,
                current_correction,
            )
        else:
            old_matrix_check = True

        # save a flag called old_matrix_check to determine if the current image was corrected using
        # (True) or not using (False) the parameter in the class

        self._correction_matrix_dict["old_matrix_check"] = old_matrix_check

        conv_img = physical_unit_data(
            img,
            angular_correction_matrix,
            e_correction,
            jacobian_determinant,
        )

        data_array = xr.DataArray(
            data=conv_img,
            coords={
                conversion_parameters["dims"][0]: angle_axis,
                conversion_parameters["dims"][1]: ek_axis,
            },
            dims=conversion_parameters["dims"],
            attrs={"conversion_parameters": conversion_parameters},
        )

        # Handle cropping based on parameters stored in correction dictionary
        crop = kwds.pop("crop", self._config.get("crop", False))
        if crop:
            try:
                range_dict: dict = self._correction_matrix_dict[lens_mode][kinetic_energy][
                    pass_energy
                ][work_function]["crop_params"]
                ang_min = range_dict["ang_min"]
                ang_max = range_dict["ang_max"]
                ek_min = range_dict["ek_min"]
                ek_max = range_dict["ek_max"]
                if self.print_msg:
                    print("Using saved crop parameters...")
                data_array = crop_xarray(data_array, ang_min, ang_max, ek_min, ek_max)
            except KeyError:
                try:
                    ang_range_min = (
                        kwds["ang_range_min"]
                        if "ang_range_min" in kwds
                        else self._config["ang_range_min"]
                    )
                    ang_range_max = (
                        kwds["ang_range_max"]
                        if "ang_range_max" in kwds
                        else self._config["ang_range_max"]
                    )
                    ek_range_min = (
                        kwds["ek_range_min"]
                        if "ek_range_min" in kwds
                        else self._config["ek_range_min"]
                    )
                    ek_range_max = (
                        kwds["ek_range_max"]
                        if "ek_range_max" in kwds
                        else self._config["ek_range_max"]
                    )
                    ang_min = (
                        ang_range_min
                        * (
                            data_array.coords[data_array.dims[0]][-1]
                            - data_array.coords[data_array.dims[0]][0]
                        )
                        + data_array.coords[data_array.dims[0]][0]
                    )
                    ang_max = (
                        ang_range_max
                        * (
                            data_array.coords[data_array.dims[0]][-1]
                            - data_array.coords[data_array.dims[0]][0]
                        )
                        + data_array.coords[data_array.dims[0]][0]
                    )
                    ek_min = (
                        ek_range_min
                        * (
                            data_array.coords[data_array.dims[1]][-1]
                            - data_array.coords[data_array.dims[1]][0]
                        )
                        + data_array.coords[data_array.dims[1]][0]
                    )
                    ek_max = (
                        ek_range_max
                        * (
                            data_array.coords[data_array.dims[1]][-1]
                            - data_array.coords[data_array.dims[1]][0]
                        )
                        + data_array.coords[data_array.dims[1]][0]
                    )
                    if self.print_msg:
                        print("Cropping parameters not found, using cropping ranges from config...")
                    data_array = crop_xarray(data_array, ang_min, ang_max, ek_min, ek_max)
                except KeyError:
                    if self.print_msg:
                        print(
                            "Warning: Cropping parameters not found, "
                            "use method crop_tool() after loading.",
                        )

        return data_array

    def crop_tool(
        self,
        raw_img: np.ndarray,
        lens_mode: str,
        kinetic_energy: float,
        pass_energy: float,
        work_function: float,
        apply: bool = False,
        **kwds,
    ):
        """Crop tool for selecting cropping parameters

        Args:
            raw_img (np.ndarray): Raw image data, numpy 2d matrix
            lens_mode (str): analzser lens mode, check calib2d for a list
                of modes CamelCase naming convention e.g. "WideAngleMode"
            kinetic_energy (float): set analyser kinetic energy
            pass_energy (float): set analyser pass energy
            work_function (float): set analyser work function
            apply (bool, optional): Option to directly apply the pre-selected cropping parameters.
                Defaults to False.
            **kwds: Keyword parameters for the crop tool:

                - ek_range_min
                - ek_range_max
                - ang_range_min
                - ang_range_max
        """
        data_array = self.convert_image(
            raw_img=raw_img,
            lens_mode=lens_mode,
            kinetic_energy=kinetic_energy,
            pass_energy=pass_energy,
            work_function=work_function,
            crop=False,
        )

        matplotlib.use("module://ipympl.backend_nbagg")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            mesh_obj = data_array.plot(ax=ax)
        except AttributeError:
            print("Load the scan first!")
            raise

        lineh1 = ax.axhline(y=data_array.Angle[0])
        lineh2 = ax.axhline(y=data_array.Angle[-1])
        linev1 = ax.axvline(x=data_array.Ekin[0])
        linev2 = ax.axvline(x=data_array.Ekin[-1])

        try:
            ang_range_min = (
                kwds["ang_range_min"] if "ang_range_min" in kwds else self._config["ang_range_min"]
            )
            ang_range_max = (
                kwds["ang_range_max"] if "ang_range_max" in kwds else self._config["ang_range_max"]
            )
            ek_range_min = (
                kwds["ek_range_min"] if "ek_range_min" in kwds else self._config["ek_range_min"]
            )
            ek_range_max = (
                kwds["ek_range_max"] if "ek_range_max" in kwds else self._config["ek_range_max"]
            )
            ang_min = (
                ang_range_min
                * (
                    data_array.coords[data_array.dims[0]][-1]
                    - data_array.coords[data_array.dims[0]][0]
                )
                + data_array.coords[data_array.dims[0]][0]
            )
            ang_max = (
                ang_range_max
                * (
                    data_array.coords[data_array.dims[0]][-1]
                    - data_array.coords[data_array.dims[0]][0]
                )
                + data_array.coords[data_array.dims[0]][0]
            )
            ek_min = (
                ek_range_min
                * (
                    data_array.coords[data_array.dims[1]][-1]
                    - data_array.coords[data_array.dims[1]][0]
                )
                + data_array.coords[data_array.dims[1]][0]
            )
            ek_max = (
                ek_range_max
                * (
                    data_array.coords[data_array.dims[1]][-1]
                    - data_array.coords[data_array.dims[1]][0]
                )
                + data_array.coords[data_array.dims[1]][0]
            )
        except KeyError:
            try:
                range_dict = self._correction_matrix_dict[lens_mode][kinetic_energy][pass_energy][
                    work_function
                ]["crop_params"]

                ek_min = range_dict["ek_min"]
                ek_max = range_dict["ek_max"]
                ang_min = range_dict["ang_min"]
                ang_max = range_dict["ang_max"]
            except KeyError:
                ek_min = data_array.coords[data_array.dims[1]][0]
                ek_max = data_array.coords[data_array.dims[1]][-1]
                ang_min = data_array.coords[data_array.dims[0]][0]
                ang_max = data_array.coords[data_array.dims[0]][-1]

        vline_range = [ek_min, ek_max]
        hline_range = [ang_min, ang_max]

        vline_slider = ipw.FloatRangeSlider(
            description="Ekin",
            value=vline_range,
            min=data_array.Ekin[0],
            max=data_array.Ekin[-1],
            step=0.01,
        )
        hline_slider = ipw.FloatRangeSlider(
            description="Angle",
            value=hline_range,
            min=data_array.Angle[0],
            max=data_array.Angle[-1],
            step=0.1,
        )
        clim_slider = ipw.FloatRangeSlider(
            description="colorbar limits",
            value=[data_array.data.min(), data_array.data.max()],
            min=data_array.data.min(),
            max=data_array.data.max(),
        )

        def update(hline, vline, v_vals):
            lineh1.set_ydata(hline[0])
            lineh2.set_ydata(hline[1])
            linev1.set_xdata(vline[0])
            linev2.set_xdata(vline[1])
            mesh_obj.set_clim(vmin=v_vals[0], vmax=v_vals[1])
            fig.canvas.draw_idle()

        ipw.interact(
            update,
            hline=hline_slider,
            vline=vline_slider,
            v_vals=clim_slider,
        )

        def cropit(val):  # pylint: disable=unused-argument
            ang_min = min(hline_slider.value)
            ang_max = max(hline_slider.value)
            ek_min = min(vline_slider.value)
            ek_max = max(vline_slider.value)
            self._data_array = crop_xarray(data_array, ang_min, ang_max, ek_min, ek_max)
            self._correction_matrix_dict[lens_mode][kinetic_energy][pass_energy][work_function] = {
                "crop_params": {
                    "ek_min": ek_min,
                    "ek_max": ek_max,
                    "ang_min": ang_min,
                    "ang_max": ang_max,
                },
            }
            self._config["ek_range_min"] = (
                (ek_min - data_array.coords[data_array.dims[1]][0])
                / (
                    data_array.coords[data_array.dims[1]][-1]
                    - data_array.coords[data_array.dims[1]][0]
                )
            ).item()
            self._config["ek_range_max"] = (
                (ek_max - data_array.coords[data_array.dims[1]][0])
                / (
                    data_array.coords[data_array.dims[1]][-1]
                    - data_array.coords[data_array.dims[1]][0]
                )
            ).item()
            self._config["ang_range_min"] = (
                (ang_min - data_array.coords[data_array.dims[0]][0])
                / (
                    data_array.coords[data_array.dims[0]][-1]
                    - data_array.coords[data_array.dims[0]][0]
                )
            ).item()
            self._config["ang_range_max"] = (
                (ang_max - data_array.coords[data_array.dims[0]][0])
                / (
                    data_array.coords[data_array.dims[0]][-1]
                    - data_array.coords[data_array.dims[0]][0]
                )
            ).item()
            self._config["crop"] = True

            ax.cla()
            self._data_array.plot(ax=ax, add_colorbar=False)
            fig.canvas.draw_idle()

            vline_slider.close()
            hline_slider.close()
            clim_slider.close()
            apply_button.close()

        apply_button = ipw.Button(description="Crop")
        display(apply_button)
        apply_button.on_click(cropit)
        plt.show()
        if apply:
            cropit("")
