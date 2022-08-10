import os
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

import specsanalyzer
from specsanalyzer import io
from specsanalyzer.convert import calculate_matrix_correction
from specsanalyzer.convert import physical_unit_data
from specsanalyzer.img_tools import fourier_filter_2D
from specsanalyzer.metadata import MetaHandler
from specsanalyzer.settings import parse_config

# from typing import Any
# from typing import List
# from typing import Tuple
# import numpy as np
# from .convert import convert_image

package_dir = os.path.dirname(specsanalyzer.__file__)


class SpecsAnalyzer:
    """[summary]"""

    def __init__(
        self,
        metadata: dict = {},
        config: Union[dict, Path, str] = {},
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

        self._correction_matrix_dict = {}

    def __repr__(self):
        if self._config is None:
            s = "No configuration available"
        else:
            s = print(self._config)
        # TODO Proper report with scan number, dimensions, configuration etc.
        return s if s is not None else ""

    def convert_image(
        self,
        raw_img: np.ndarray,
        pass_energy: float,
        kinetic_energy: float,
        lens_mode: str,
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
                img = fourier_filter_2D(raw_img, fft_filter_peaks)
            except KeyError:
                img = raw_img
        else:
            img = raw_img

        # TODO check valid lens modes

        try:
            ek_axis = self._correction_matrix_dict[lens_mode][pass_energy][
                kinetic_energy
            ]["ek_axis"]
            angle_axis = self._correction_matrix_dict[lens_mode][pass_energy][
                kinetic_energy
            ]["angle_axis"]
            angular_correction_matrix = self._correction_matrix_dict[
                lens_mode
            ][pass_energy][kinetic_energy]["angular_correction_matrix"]
            e_correction = self._correction_matrix_dict[lens_mode][
                pass_energy
            ][kinetic_energy]["e_correction"]
            jacobian_determinant = self._correction_matrix_dict[lens_mode][
                pass_energy
            ][kinetic_energy]["jacobian_determinant"]
        except KeyError:
            (
                ek_axis,
                angle_axis,
                angular_correction_matrix,
                e_correction,
                jacobian_determinant,
            ) = calculate_matrix_correction(
                lens_mode,
                pass_energy,
                kinetic_energy,
                binning,
                self._config,
            )
            # TODO: make this function compatible, call the function
            # calculate_polynomial_coef_da inside.
            # TODO: store result in dictionary.

        conv_img = physical_unit_data(
            img,
            angular_correction_matrix,
            e_correction,
            jacobian_determinant,
        )
        # TODO: make function compatible, check interpolation functions.
        # TODO generate xarray
        # TODO: annotate with metadata
        da = xr.DataArray(
            data=conv_img,
            coords={ek_axis, angle_axis},
            dims=list("Ekin", "Angle"),
        )

        return da
