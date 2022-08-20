"""This is a code that performs several tests for the input/output functions
"""
import random

import numpy as np
import pytest
import xarray as xr

from specsanalyzer.io import load_h5
from specsanalyzer.io import to_h5
from tests.helpers import simulate_binned_data

shapes = []
for n in range(4):
    shapes.append(tuple(np.random.randint(10) + 2 for i in range(n + 1)))
axes_names = ["X", "Y", "T", "E"]
random.shuffle(axes_names)
binned_arrays = [simulate_binned_data(s, axes_names[: len(s)]) for s in shapes]


@pytest.mark.parametrize(
    "_da",
    binned_arrays,
    ids=lambda x: f"ndims:{len(x.shape)}",
)
def test_save_and_load_hdf5(_da):
    """Test the hdf5 saving/loading function."""
    faddr = "test.h5"
    to_h5(_da, faddr, mode="w")
    loaded = load_h5(faddr)
    xr.testing.assert_equal(_da, loaded)
