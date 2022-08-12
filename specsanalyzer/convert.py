import numpy as np
import xarray as xr


def convert_image(
    raw_image: np.ndarray,
    pass_energy: float,
    kinetic_energy: float,
    lens_mode: int,
    binning: int,
    calibration_dict: dict = {},
    detector_voltage: float = np.NaN,
) -> xr.DataArray:
    """Converts raw image into physical unit coordinates.

    Args:
        ....

    Returns:
        ...
    """


# TODO: populate
