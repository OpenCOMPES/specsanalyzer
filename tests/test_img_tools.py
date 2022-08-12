import numpy as np

from specsanalyzer.img_tools import fourier_filter_2d

bins2d = (95, 34)
array2d = np.random.normal(size=bins2d)
# strip negative values
for i in range(0, array2d.shape[0]):
    for j in range(0, array2d.shape[1]):
        array2d[i, j] = array2d[i][j] if array2d[i][j] > 0 else 0


def test_fourier_filter_2d():
    np.testing.assert_allclose(
        array2d,
        fourier_filter_2d(array2d, []),
        atol=1e-10,
    )


def test_fourier_filter_2d_raises():
    with np.testing.assert_raises(KeyError):
        fourier_filter_2d(array2d, [{"amplitude": 1}])
