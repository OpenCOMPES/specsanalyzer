"""This is the specsanalyzer core class

"""
import os
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import xarray as xr

from specsanalyzer import io
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer.convert import physical_unit_data
from specsanalyzer.img_tools import crop_xarray
from specsanalyzer.img_tools import fourier_filter_2d
from specsanalyzer.metadata import MetaHandler
from specsanalyzer.settings import parse_config

# from typing import Any
# from typing import List
# from typing import Tuple
# import numpy as np
# from .convert import convert_image

package_dir = os.path.dirname(__file__)


class SpecsAnalyzer:  # pylint: disable=dangerous-default-value
    """[summary]"""

    def __init__(
        self,
        metadata: Dict[Any, Any] = {},
        config: Union[Dict[Any, Any], str] = {},
    ):

        self._config = parse_config(config)

        try:
            self._config["calib2d_dict"] = io.parse_calib2d_to_dict(
                self._config["calib2d_file"],
            )
        except FileNotFoundError:  # default location relative to package directory
            self._config["calib2d_dict"] = io.parse_calib2d_to_dict(
                os.path.join(package_dir, self._config["calib2d_file"]),
            )

        self._attributes = MetaHandler(meta=metadata)

        self._correction_matrix_dict: Dict[Any, Any] = {}

    def __repr__(self):
        if self._config is None:
            pretty_str = "No configuration available"
        else:
            for key in self._config:
                pretty_str += print(f"{self._config[key]}\n")
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

    def convert_image(
        self,
        raw_img: np.ndarray,
        lens_mode: str,
        kinetic_energy: float,
        pass_energy: float,
        work_function: float,
        **kwds,
    ) -> xr.DataArray:
        """Converts raw image into physical unit coordinates.
        Args:
            raw_img: raw image data as numpy 2D ndarray
            pass_energy: the pass energy in eV
            kinetic_energy: the kinetic energy in eV
            lens_mode: the lens mode as string. Depending on the calibration file,
                    the following lens modes are supported:
                    -LowAngularDispersion
                    -MediumAngularDispersion
                    -HighAngularDispersion
                    -WideAngleMode
                    -LargeArea
                    -MediumArea
                    -SmallArea
                    -SmallArea2
                    -HighMagnification2
                    -HighMagnification
                    -LowMagnification
                    -SuperWideAngleMode
            **kwds: additional config keywords

        Raises:
            ...

        Returns:
            da: xarray DataArray object with kinetic energy and angle/position as
                coordinates
        """

        apply_fft_filter = kwds.pop(
            "apply_fft_filter",
            self._config["apply_fft_filter"],
        )
        binning = kwds.pop("binning", self._config["binning"])

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

        # TODO check valid lens modes
        # create a tuple containing the current scan parameters

        try:
            # check if the config file contains the last scan parameters
            # old_params in the current lens_mode
            # this contains 3 element tuples of the form
            # [kinetic_energy, pass_energy, work_function]

            old_db = self._config["calib2d_dict"][lens_mode][kinetic_energy][
                pass_energy
            ][work_function]

            ek_axis = old_db["ek_axis"]
            angle_axis = old_db["angle_axis"]
            angular_correction_matrix = old_db["angular_correction_matrix"]
            e_correction = old_db["e_correction"]
            jacobian_determinant = old_db["jacobian_determinant"]

        except KeyError:
            old_matrix_check = False
            (
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
            self._config["calib2d_dict"][lens_mode][kinetic_energy] = {
                pass_energy: {
                    work_function: {
                        "ek_axis": ek_axis,
                        "angle_axis": angle_axis,
                        "angular_correction_matrix": angular_correction_matrix,
                        "e_correction": e_correction,
                        "jacobian_determinant": jacobian_determinant,
                    },
                },
            }

            # TODO: make this function compatible, call the function
            # calculate_polynomial_coef_da inside.

        else:
            old_matrix_check = True
            # print("Old correction matrix")
            # print(last_scan)

        # save a flag called old_matrix_check to determine if the current
        # image was corrected using (True) or not using (False) the
        # parameter in the class

        self._config["calib2d_dict"]["old_matrix_check"] = old_matrix_check

        conv_img = physical_unit_data(
            img,
            angular_correction_matrix,
            e_correction,
            jacobian_determinant,
        )

        # TODO: annotate with metadata
        da = xr.DataArray(
            data=conv_img,
            coords={"Angle": angle_axis, "Ekin": ek_axis},
            dims=["Angle", "Ekin"],
        )

        # TODO discuss how to handle cropping. Can he store one set of cropping
        # parameters in the config, or should we store one set per pass energy/
        # lens mode/ kinetic energy in the dict?

        crop = kwds.pop("crop", self._config["crop"])
        if crop:
            ek_min = kwds.pop("ek_min", self._config["ek_min"])
            ek_max = kwds.pop("ek_max", self._config["ek_max"])
            ang_min = kwds.pop("ang_min", self._config["ang_min"])
            ang_max = kwds.pop("ang_max", self._config["ang_max"])
            da = crop_xarray(da, ang_min, ang_max, ek_min, ek_max)

        return da
