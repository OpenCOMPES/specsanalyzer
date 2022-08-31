"""This module contains image manipulation tools for the specsanalyzer package

"""
from typing import Sequence

import numpy as np
import xarray as xr


def gauss2d(
    # pylint: disable=invalid-name, too-many-arguments
    x: float,
    y: float,
    mx: float,
    my: float,
    sx: float,
    sy: float,
) -> float:
    """Function to calculate a 2-dimensional Gaussian peak function without
       correlation, and amplitude 1.

    Args:
        x: independent x-variable
        y: independent y-variable
        mx: x-center of the 2D Gaussian
        my: y-center of the 2D Gaussian
        sx: Sigma in y direction
        sy: Sigma in x direction

    Returns:
        peak intensity at the given (x, y) coordinates.
    """

    return np.exp(
        -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0)),
    )


def fourier_filter_2d(
    image: np.ndarray,
    peaks: Sequence,
    ret: str = "filtered",
) -> np.ndarray:
    """Function to Fourier filter an image for removal of regular pattern artefacts,
       e.g. grid lines.

    Args:
        image: the input image
        peaks: list of dicts containing the following information about a "peak" in the
               Fourier image:
               'pos_x', 'pos_y', sigma_x', sigma_y', 'amplitude'. Define one entry for
               each feature you want to suppress in the Fourier image, where amplitude
               1 corresponds to full suppression.
        ret: flag to indicate which data to return. Possible values are:
             'filtered', 'fft', 'mask', 'filtered_fft'

    Returns:
        The chosen image data. Default is the filtered real image.
    """

    # Do Fourier Transform of the (real-valued) image
    image_fft = np.fft.rfft2(image)
    mask = np.ones(image_fft.shape)

    for i in range(image_fft.shape[0]):
        for j in range(image_fft.shape[1]):
            for peak in peaks:
                try:
                    mask[i][j] -= peak["amplitude"] * gauss2d(
                        i,
                        j,
                        peak["pos_x"],
                        peak["pos_y"],
                        peak["sigma_x"],
                        peak["sigma_y"],
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"The peaks input is supposed to be a list of dicts with the\
following structure: pos_x, pos_y, sigma_x, sigma_y, amplitude. The error was {exc}.",
                    ) from exc

    # apply mask to the FFT, and transform back
    filtered = np.fft.irfft2(image_fft * mask)

    # strip negative values
    for i in range(0, filtered.shape[0]):
        for j in range(0, filtered.shape[1]):
            filtered[i, j] = filtered[i][j] if filtered[i][j] > 0 else 0

    if ret == "filtered":
        return filtered
    if ret == "fft":
        return image_fft
    if ret == "mask":
        return mask
    if ret == "filtered_fft":
        return image_fft * mask
    return filtered  # default return


def crop_xarray(
    data_array: xr.DataArray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> xr.DataArray:
    """Crops an xarray according to the provided coordinate boundaries.

    Args:
        data_array: the input xarray DataArray
        x_min: the minimum position along the first element in the x-array dims list.
        x_max: the maximum position along the first element in the x-array dims list.
        y_min: the minimum position along the second element in the x-array dims list.
        y_max: the maximum position along the second element in the x-array dims list.

    Returns:
        The cropped xarray DataArray.
    """

    x_axis = data_array.coords[data_array.dims[0]]
    y_axis = data_array.coords[data_array.dims[1]]
    x_mask = (x_axis >= x_min) & (x_axis <= x_max)
    y_mask = (y_axis >= y_min) & (y_axis <= y_max)
    data_array_cropped = data_array.where(x_mask & y_mask, drop=True)

    return data_array_cropped
