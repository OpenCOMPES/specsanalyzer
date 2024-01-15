"""This module contains image manipulation tools for the specsanalyzer package

"""
from typing import Sequence
from typing import Union

from matplotlib import lines
import numpy as np
import xarray as xr


class DraggableLines:
    """
    Class to run an interactive tool to drag lines over an image
    an store the last positions. Used by the cropping tool in
    specsscan.load_scan() method.
    """
    def __init__(self, ax, fig, kind, range_dict, xarray):
        self.ax = ax
        self.c = ax.get_figure().canvas
        self.o = kind
        self.xory = range_dict[kind]['val']
        self.xarray = xarray
        self.follower = None
        self.releaser = None
        dims = xarray.dims
        props = {"boxstyle": 'round', "facecolor": 'white', "alpha": 0.5}
        self.text = ax.text(
            range_dict[kind]['x'],
            range_dict[kind]['y'],
            f"{self.o} " + f"{self.xory:.2f}",
            transform=fig.transFigure,
            bbox=props
        )

        if kind in ("Ang1", "Ang2"):
            self.x = xarray[f"{dims[1]}"].data
            self.y = [range_dict[kind]['val']] * len(self.x)

        elif kind in ("Ek1", "Ek2"):
            self.y = xarray[f"{dims[0]}"].data
            self.x = [range_dict[kind]['val']] * len(self.y)

        self.line = lines.Line2D(self.x, self.y, picker=5)
        self.ax.add_line(self.line)
        self.c.draw_idle()

        self.sid = self.c.mpl_connect('button_press_event', self.clickonline)

    def clickonline(self, event):
        """
        Checks if the line clicked belongs to this instance.
        """
        if event.inaxes != self.line.axes:
            return
        contains = self.line.contains(event)
        if not contains[0]:
            return

        self.follower = self.c.mpl_connect("motion_notify_event", self.followmouse)
        self.releaser = self.c.mpl_connect("button_release_event", self.releaseonclick)

    def followmouse(self, event):
        """
        Sets the selected line position to the mouse position,
        while updating the text box in real time.
        """
        if self.o in ("Ang1", "Ang2"):
            if event.ydata:
                self.line.set_ydata([event.ydata] * len(self.x))
            else:
                self.line.set_ydata([event.ydata] * len(self.x))
            self.xory = self.line.get_ydata()[0]
            self.text.set_text(f"{self.o} " + f"{self.xory:.2f}")

        elif self.o in ("Ek1", "Ek2"):
            self.line.set_xdata([event.xdata] * len(self.y))
            self.xory = self.line.get_xdata()[0]
            self.text.set_text(f"{self.o} " + f"{self.xory:.2f}")

        self.c.draw_idle()

    def releaseonclick(self, event):  # pylint: disable=unused-argument
        """
        Disconnects the interaction on mouse release.
        """
        self.c.draw_idle()
        self.c.mpl_disconnect(self.releaser)
        self.c.mpl_disconnect(self.follower)


def gauss2d(
    # pylint: disable=invalid-name, too-many-arguments
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    mx: float,
    my: float,
    sx: float,
    sy: float,
) -> Union[float, np.ndarray]:
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
    xgrid, ygrid = np.meshgrid(
        range(image_fft.shape[0]),
        range(image_fft.shape[1]),
        indexing="ij",
        sparse=True,
    )
    for peak in peaks:
        try:
            mask -= peak["amplitude"] * gauss2d(
                xgrid,
                ygrid,
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
    filtered = filtered.clip(min=0)
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
