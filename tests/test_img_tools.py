import numpy as np

from specsanalyzer.img_tools import fourier_filter_2D

bins2d = (95, 34)
array2d = np.random.normal(size=bins2d)


def test_fourier_filter_2D():
    np.testing.assert_allclose(array2d, fourier_filter_2D(array2d, []))


def test_fourier_filter_2D_raises():
    with np.testing.assert_raises(KeyError):
        fourier_filter_2D(array2d, [{"amplitude": 1}])
