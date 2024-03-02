"""Helper functions for tests"""

import numpy as np
import xarray as xr

UNITS = {"X": "mm", "Y": "mm", "T": "fs", "E": "eV"}


def simulate_binned_data(shape: tuple, dims: list):
    """Generate a fake xr.DataArray as those generated by binning

    Used for testing purpouses

    Args:
        shape: Shape ouf the data
        dims: name of the dimensions

    Returns:
        the simulated data array
    """
    assert len(dims) == len(
        shape,
    ), "number of dimesions and data shape must coincide"

    ret = xr.DataArray(
        data=np.random.rand(*shape),
        coords={d: np.linspace(-1, 1, s) for d, s in zip(dims, shape)},
        attrs={
            "metadata": {
                "list": [1, 2, 3],
                "string": "asdf",
                "int": 1,
                "float": 1.0,
                "bool": True,
                "nested": {
                    "nestedentry": 1.0,
                },
            },
            "units": "Counts",
        },
    )
    for dim in dims:
        ret[dim].attrs["unit"] = UNITS[dim]

    return ret


def get_linear_bin_edges(array: np.ndarray) -> np.ndarray:
    """returns the bin edges of the given array

    Args:
        array: the array of N center values from which to evaluate the bin range.
         Must be linear.

    Returns:
        edges: array of edges, with shape N+1
    """
    step = array[1] - array[0]
    last_step = array[-2] - array[-3]
    assert np.allclose(last_step, step), "not a linear array"
    return np.linspace(
        array[0],
        array[-1],
        array.size + 1,
    )
