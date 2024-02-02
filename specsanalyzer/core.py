"""This is the specsanalyzer core class

"""
import os
from typing import Any
from typing import Dict
from typing import Generator
from typing import Tuple
from typing import Union

import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display

from specsanalyzer import io
from specsanalyzer.config import parse_config
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer.convert import physical_unit_data
from specsanalyzer.img_tools import crop_xarray
from specsanalyzer.img_tools import fourier_filter_2d
from specsanalyzer.metadata import MetaHandler

package_dir = os.path.dirname(__file__)


class SpecsAnalyzer:  # pylint: disable=dangerous-default-value
    """[summary]"""

    def __init__(
        self,
        metadata: Dict[Any, Any] = {},
        config: Union[Dict[Any, Any], str] = {},
        **kwds,
    ):

        self._config = parse_config(
            config,
            **kwds,
        )
        self._attributes = MetaHandler(meta=metadata)
        self._data_array = None
        self.print_msg = True
        try:
            self._config["calib2d_dict"] = io.parse_calib2d_to_dict(
                self._config["calib2d_file"],
            )

        except FileNotFoundError:  # default location relative to package directory
            self._config["calib2d_dict"] = io.parse_calib2d_to_dict(
                os.path.join(package_dir, self._config["calib2d_file"]),
            )

        self._correction_matrix_dict: Dict[Any, Any] = {}

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

    @property
    def correction_matrix_dict(self):
        """Get correction_matrix_dict"""
        return self._correction_matrix_dict

    def convert_image(
        self,
        raw_img: np.ndarray,
        lens_mode: str,
        kinetic_energy: float,
        pass_energy: float,
        work_function: float,
        **kwds,
    ) -> xr.DataArray:
        """Converts an imagin in physical unit data, angle vs energy


        Args:
            raw_img (np.ndarray): Raw image data, numpy 2d matrix
            lens_mode (str):
                analzser lens mode, check calib2d for a list
                of modes Camelback naming convention e.g. "WideAngleMode"

            kinetic_energy (float): set analyser kinetic energy
            pass_energy (float): set analyser pass energy
            work_function (float): set analyser work function

        Returns:
            xr.DataArray: xarray containg the corrected data and kinetic
            and angle axis
        """

        apply_fft_filter = kwds.pop(
            "apply_fft_filter",
            self._config.get("apply_fft_filter", False),
        )
        binning = kwds.pop("binning", self._config.get("binning", 1))

        if apply_fft_filter:
            try:
                fft_filter_peaks = kwds.pop(
                    "fft_filter_peaks",
                    self._config["fft_filter_peaks"],
                )
                img = fourier_filter_2d(raw_img, fft_filter_peaks)
            except KeyError:
                img = raw_img
        else:
            img = raw_img

        # TODO add image rotation

        # look for the lens mode in the dictionary
        try:
            supported_angle_modes = self._config["calib2d_dict"]["supported_angle_modes"]
            supported_space_modes = self._config["calib2d_dict"]["supported_space_modes"]
        # pylint: disable=duplicate-code
        except KeyError as exc:
            raise KeyError(
                "The supported modes were not found in the calib2d dictionary",
            ) from exc

        if lens_mode not in [*supported_angle_modes, *supported_space_modes]:
            raise ValueError(
                f"convert_image: unsupported lens mode: '{lens_mode}'",
            )

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
            old_matrix_check = False
            (  # pylint: disable=duplicate-code
                ek_axis,
                angle_axis,
                angular_correction_matrix,
                e_correction,
                jacobian_determinant,
            ) = calculate_matrix_correction(
                lens_mode,
                kinetic_energy,
                pass_energy,
                work_function,
                binning,
                self._config,
            )

            # save the config parameters for later use
            # collect the info in a new nested dictionary
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
            self._correction_matrix_dict = dict(
                mergedicts(self._correction_matrix_dict, current_correction),
            )

        else:
            old_matrix_check = True

        # save a flag called old_matrix_check to determine if the current
        # image was corrected using (True) or not using (False) the
        # parameter in the class

        self._correction_matrix_dict["old_matrix_check"] = old_matrix_check

        conv_img = physical_unit_data(
            img,
            angular_correction_matrix,
            e_correction,
            jacobian_determinant,
        )

        # TODO: annotate with metadata

        if lens_mode in supported_angle_modes:
            data_array = xr.DataArray(
                data=conv_img,
                coords={"Angle": angle_axis, "Ekin": ek_axis},
                dims=["Angle", "Ekin"],
            )
        elif lens_mode in supported_space_modes:
            data_array = xr.DataArray(
                data=conv_img,
                coords={"Position": angle_axis, "Ekin": ek_axis},
                dims=["Position", "Ekin"],
            )

        # TODO discuss how to handle cropping. Can he store one set of cropping
        # parameters in the config, or should we store one set per pass energy/
        # lens mode/ kinetic energy in the dict?
        saved_rangedict = False
        crop = kwds.pop("crop", self._config.get("crop", False))
        if crop:
            try:
                range_dict: dict = self._correction_matrix_dict[lens_mode][kinetic_energy][
                    pass_energy
                ][work_function]["crop_params"]
            except KeyError:
                if self.print_msg:
                    print(
                        "Warning: Cropping parameters not found, "
                        "use method crop_tool() after loading.",
                    )
                    self.print_msg = False
            else:
                saved_rangedict = True

        if saved_rangedict:
            if self.print_msg:
                print("Using existing crop parameters...")
                self.print_msg = False
            ang_min = min(range_dict["Ang1"]["val"], range_dict["Ang2"]["val"])
            ang_max = max(range_dict["Ang1"]["val"], range_dict["Ang2"]["val"])
            ek_min = min(range_dict["Ek1"]["val"], range_dict["Ek2"]["val"])
            ek_max = max(range_dict["Ek1"]["val"], range_dict["Ek2"]["val"])
            data_array = crop_xarray(data_array, ang_min, ang_max, ek_min, ek_max)

        return data_array

    def crop_tool(
        self,
        res_xarray: xr.DataArray,
        scan_info_dict: dict,
    ):
        """Crop tool
        Args:
            res_xarray: xarray obtained from the converted raw data
            scan_info_dict: dict containing the contents of info.txt file
        """

        matplotlib.use("module://ipympl.backend_nbagg")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            if len(res_xarray.dims) == 3:
                res_xarray[:, :, 0].plot(ax=ax)
            else:  # dim == 2
                res_xarray.plot(ax=ax)
        except AttributeError:
            print("Load the scan first!")
            raise

        vline = ipw.FloatRangeSlider(
            value=[res_xarray.Ekin[0], res_xarray.Ekin[-1]],
            min=res_xarray.Ekin[0],
            max=res_xarray.Ekin[-1],
            step=0.01,
        )
        tline = ipw.FloatRangeSlider(
            value=[res_xarray.Angle[0], res_xarray.Angle[-1]],
            min=res_xarray.Angle[0],
            max=res_xarray.Angle[-1],
            step=0.01,
        )

        def update(tline, vline):  # pylint: disable=unused-argument
            return

        ipw.interact(
            update,
            tline=tline,
            vline=vline,
        )

        def cropit(val):  # pylint: disable=unused-argument
            ang_min = min(tline.value)
            ang_max = max(tline.value)
            ek_min = min(vline.value)
            ek_max = max(vline.value)
            self._data_array = crop_xarray(res_xarray, ang_min, ang_max, ek_min, ek_max)
            self._correction_matrix_dict[scan_info_dict["LensMode"]][
                scan_info_dict["KineticEnergy"]
            ][scan_info_dict["PassEnergy"]][scan_info_dict["WorkFunction"]] = {
                "crop_params": {
                    "Ek1": {"x": 0.15, "y": 0.9, "val": ek_min},
                    "Ek2": {"x": 0.30, "y": 0.9, "val": ek_max},
                    "Ang1": {"x": 0.45, "y": 0.9, "val": ang_min},
                    "Ang2": {"x": 0.60, "y": 0.9, "val": ang_max},
                },
            }
            ax.cla()
            self._data_array.plot(ax=ax, add_colorbar=False)
            fig.canvas.draw_idle()

            vline.close()
            tline.close()
            apply_button.close()

        apply_button = ipw.Button(description="Crop")
        display(apply_button)
        apply_button.on_click(cropit)
        plt.show()


def mergedicts(
    dict1: dict,
    dict2: dict,
) -> Generator[Tuple[Any, Any], None, None]:
    """Merge two dictionaries, overwriting only existing values and retaining
    previously present values

    Args:
        dict1 (dict): dictionary 1
        dict2 (dict): dictiontary 2

    Yields:
        dict: merged dictionary generator
    """
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(mergedicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict,
                #  you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception
                # raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])
