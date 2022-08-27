from pyexpat.errors import XML_ERROR_UNKNOWN_ENCODING

import numba as nb
import numpy as np
import xarray as xr
from interpolation.splines import CGrid
from interpolation.splines import eval_linear
from interpolation.splines import nodes
from interpolation.splines import UCGrid
from numba import jit
from numba import prange
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates


def get_damatrix_fromcalib2d(
    lens_mode,
    kinetic_energy,
    pass_energy,
    config_dict,
):
    """This function returns a matrix of coefficients

    Args:
        infofilename (_type_): _description_
        calib2dfilename (_type_): _description_

    Returns:
        _type_: _description_
    """
    # retardation ratio

    #
    #
    work_function = config_dict["work_function"]

    rr = (kinetic_energy - work_function) / pass_energy

    # now we have a dilemma: in igor the rr was calculated including the work function,
    # depending on the settings, one might have or not have the work function included in the
    # labview data acquisition..

    # given the lens mode get all the retardatio ratios available
    rr_vec, damatrix_full = get_rr_da(lens_mode, config_dict)
    closest_rr_index = bisection(rr_vec, rr)

    # return as the closest rr index the smallest in case of -1 output
    if closest_rr_index == -1:
        closest_rr_index = 0
    # print("closest_rr_index= ", closest_rr_index)
    # now compare the distance with the neighboring indexes,
    # we need the second nearest rr
    second_closest_rr_index = second_closest_rr(rr, rr_vec, closest_rr_index)

    # compute the rr_factor, in igor done by a BinarySearchInterp
    # find closest retardation ratio in table
    # rr_inf=BinarySearch(w_rr, rr)
    # fraction from this to next retardation ratio in table
    # rr_factor=BinarySearchInterp(w_rr, rr)-rr_inf
    rr_index = np.arange(0, rr_vec.shape[0], 1)
    rr_factor = np.interp(rr, rr_vec, rr_index) - closest_rr_index

    # print("rr_factor= ", rr_factor)

    damatrix_close = damatrix_full[closest_rr_index][:][:]
    damatrix_second = damatrix_full[second_closest_rr_index][:][:]
    # print(damatrix_close.shape)
    # print(damatrix_second.shape)

    one_mat = np.ones(damatrix_close.shape)
    rr_factor_mat = np.ones(damatrix_close.shape) * rr_factor
    damatrix = (
        damatrix_close * (one_mat - rr_factor_mat)
        + damatrix_second * rr_factor_mat
    )
    aInner = damatrix[0][0]
    damatrix = damatrix[1:][:]

    return aInner, damatrix


# Auxiliary function to load the info file

# Auxiliary function to find the closest rr index
# from https://stackoverflow.com/questions/2566412/
# find-nearest-value-in-numpy-array


def bisection(array, value):
    """Given an ``array`` , and given a ``value`` , returns an index
    j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic
    increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    This should mimick the function BinarySearch in igor pro 6"""

    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n - 1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl


def second_closest_rr(rr, rrvec, closest_rr_index):
    """_summary_

    Args:
        rr (_type_): _description_
        rrvec (_type_): _description_
        closest_rr_index (_type_): _description_

    Returns:
        _type_: _description_
    """

    # commented, modified to correctly match igor bahaviour
    """ if closest_rr_index == 0:
        second_closest_rr_index = 1
    else:
        if closest_rr_index == (rrvec.size-1):
            second_closest_rr_index = closest_rr_index-1
        else:
            # we are not at the edges, compare the neighbors and get the
            # closest

            # modified to exatly match igor behaviour,

            if (rr < rrvec[closest_rr_index]):
                second_closest_rr_index = closest_rr_index-1
            else:
                second_closest_rr_index = closest_rr_index+1 """

    if closest_rr_index == (rrvec.size - 1):
        # we are the edge: the behaviour is to not change the index
        second_closest_rr_index = closest_rr_index
    else:
        second_closest_rr_index = closest_rr_index + 1

    return second_closest_rr_index


# this function should get both the rr array, and the corresponding Da matrices
# for a certain Angle mode


def get_rr_da(lens_mode, config_dict):

    rr_array = np.array(list(config_dict["calib2d_dict"][lens_mode]["rr"]))

    dim1 = rr_array.shape[0]
    base_dict = config_dict["calib2d_dict"][lens_mode]["rr"]
    dim2 = len(base_dict[rr_array[0]])

    try:
        dim3 = len(base_dict[rr_array[0]]["Da1"])
    except KeyError:
        raise ("Da values do not exist for the given mode.")

    da_matrix = np.zeros([dim1, dim2, dim3])
    for i in range(len(rr_array)):
        aInner = base_dict[rr_array[i]]["aInner"]
        da_block = np.concatenate(
            tuple(
                [v] for k, v in base_dict[rr_array[i]].items() if k != "aInner"
            ),
        )
        da_matrix[i] = np.concatenate((np.array([[aInner] * dim3]), da_block))
    return rr_array, da_matrix


def calculate_polynomial_coef_da(
    da_matrix,
    kinetic_energy,
    pass_energy,
    eshift,
):
    """Given the da coeffiecients contained in the
    scanpareters, the program calculate the energy range based
    on the eshift parameter and fits a second order polinomial
    to the tabulated values. The polinomial coefficients
    are packed in the dapolymatrix array (row0 da1, row1 da3, ..)
    The dapolymatrix is also saved in the scanparameters dictionary

    Args:
        scanparameters (_dict_): scan parameter dictionary

    Returns:
        _np.array_: dapolymatrix, a matrix with polinomial
    """

    # get the Das from the damatrix
    # da1=currentdamatrix[0][:]
    # da3=currentdamatrix[1][:]
    # da5=currentdamatrix[2][:]
    # da7=currentdamatrix[3][:]

    # calcualte the energy values for each da, given the eshift
    da_energy = eshift * pass_energy + kinetic_energy * np.ones(eshift.shape)

    # create the polinomial coeffiecient matrix,
    # each is a second order polinomial

    dapolymatrix = np.zeros(da_matrix.shape)

    for i in range(0, da_matrix.shape[0]):
        # igor uses the fit poly 3, which should be a parabola
        dapolymatrix[i][:] = np.polyfit(
            da_energy,
            da_matrix[i][:],
            2,
        ).transpose()

    # scanparameters['dapolymatrix'] = dapolymatrix
    return dapolymatrix


# the function now returns a matrix of the fit coeffiecients,
# given the physical energy scale
# each line of the matrix is a set of coefficients for each of the
# dai corrections


def zinner(ek, angle, dapolymatrix):
    """_summary_

    Args:
        ek (_type_): _description_
        angle (_type_): _description_
        dapolymatrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    # poly(D1, Ek )*(Ang) + 10^-2*poly(D3, Ek )*(Ang)^3 +
    # 10^-4*poly(D5, Ek )*(Ang)^5 + 10^-6*poly(D7, Ek )*(Ang)^7
    out = 0

    for i in np.arange(0, dapolymatrix.shape[0], 1):
        out = out + (
            (10.0 ** (-2 * i))
            * (angle ** (1 + 2 * i))
            * np.polyval(dapolymatrix[i][:], ek)
        )
    return out


def zinner_diff(ek, angle, dapolymatrix):
    """_summary_ poly(D1, Ek ) + 3*10^-2*poly(D3, Ek )*(Ang)^2
    + 5*10^-4*poly(D5, Ek )*(Ang)^4 + 7*10^-6*poly(D7, Ek )*(Ang)^6
    Args:
        ek (_type_): _description_
        angle (_type_): _description_
        dapolymatrix (_type_): _description_

    Returns:
        _type_: _description_
    """

    out = 0

    for i in np.arange(0, dapolymatrix.shape[0], 1):

        out = out + (
            (10.0 ** (-2 * i))
            * (1 + 2 * i)
            * (angle ** (2 * i))
            * np.polyval(dapolymatrix[i][:], ek)
        )

    return out


def mcp_position_mm(ek, angle, a_inner, dapolymatrix):
    """_summary_

    Args:
        ek (_type_): _description_
        angle (_type_): _description_
        scanparameters (_type_): _description_

    Returns:
        _type_: _description_
    """

    mask = np.less_equal(np.abs(angle), a_inner)

    a_inner_vec = np.ones(angle.shape) * a_inner

    result = np.where(
        mask,
        zinner(ek, angle, dapolymatrix),
        np.sign(angle)
        * (
            zinner(ek, a_inner_vec, dapolymatrix)
            + (np.abs(angle) - a_inner_vec)
            * zinner_diff(ek, a_inner_vec, dapolymatrix)
        ),
    )
    return result


def calculate_matrix_correction(
    lens_mode,
    pass_energy,
    kinetic_energy,
    binning,
    config_dict,
):
    """_summary_

    Args:
        scanparameters (_type_): _description_

    Returns:
        _type_: _description_
    """

    eshift = np.array(config_dict["calib2d_dict"]["eShift"])

    aInner, damatrix = get_damatrix_fromcalib2d(
        lens_mode,
        kinetic_energy,
        pass_energy,
        config_dict,
    )

    dapolymatrix = calculate_polynomial_coef_da(
        damatrix,
        kinetic_energy,
        pass_energy,
        eshift,
    )

    # de1 = [config_dict["calib2d_dict"]["De1"]]
    de1 = [config_dict["calib2d_dict"]["De1"]]
    erange = config_dict["calib2d_dict"]["eRange"]
    arange = config_dict["calib2d_dict"][lens_mode]["default"]["aRange"]
    nx_pixel = config_dict["nx_pixel"]
    ny_pixel = config_dict["ny_pixel"]
    pixelsize = config_dict["pixel_size"]
    # binning = float(scanparameters["Binning"])*2
    magnification = config_dict["magnification"]

    nx_bins = int(nx_pixel / binning)
    ny_bins = int(ny_pixel / binning)
    ek_low = kinetic_energy + erange[0] * pass_energy
    ek_high = kinetic_energy + erange[1] * pass_energy

    # in igor we have
    # setscale/P x, EkinLow, (EkinHigh-EkinLow)/nx_pixel, "eV",
    # setscale/P y, AzimuthLow*1.2, (AzimuthHigh-AzimuthLow)*1.2/ny_pixel

    # is the behaviour of arange more similar??

    # assume an even number of pixels on the detector, seems reasonable
    ek_axis = np.linspace(ek_low, ek_high, nx_bins, endpoint=False)
    # ek_axis = np.arange(ek_low, ek_high, ek_step)
    # we need the arange as well as 2d array
    # arange was defined in the igor procedure Calculate_Da_values
    # it seems to be a constant, written in the calib2d file header
    # I decided to rename from "AzimuthLow"
    angle_low = arange[0] * 1.2
    angle_high = arange[1] * 1.2
    # angle_step=(angle_high-angle_low)/ny_bins

    # check the effect of the additional range x1.2;
    # this is present in the igor code

    angle_axis = np.linspace(angle_low, angle_high, ny_bins, endpoint=False)
    # angle_axis = np.arange(angle_low, angle_high, angle_step)

    # the original program defines 2 waves,
    mcp_position_mm_matrix = np.zeros([nx_bins, ny_bins])
    angular_correction_matrix = np.zeros([nx_bins, ny_bins])
    e_correction = np.zeros(ek_axis.shape)
    # let's create a meshgrid for vectorized evaluation
    ek_mesh, angle_mesh = np.meshgrid(ek_axis, angle_axis)
    mcp_position_mm_matrix = mcp_position_mm(
        ek_mesh,
        angle_mesh,
        aInner,
        dapolymatrix,
    )

    Ang_Offset_px = config_dict["Ang_Offset_px"]
    E_Offset_px = config_dict["E_Offset_px"]

    angular_correction_matrix = (
        mcp_position_mm_matrix / magnification / (pixelsize * binning)
        + ny_bins / 2
        + Ang_Offset_px
    )

    e_correction = (
        (ek_axis - kinetic_energy * np.ones(ek_axis.shape))
        / pass_energy
        / de1
        / magnification
        / (pixelsize * binning)
        + nx_bins / 2
        + E_Offset_px
    )

    # print(1 / pass_energy/float(de1[0])/magnification/(pixelsize*binning))
    # print("pass_energy: ", pass_energy,
    # "de1: ", de1,
    # "magnification: ", magnification,
    # "pixelsize: ", pixelsize,
    # "binning: ", binning)
    # calculate Jacobian determinant

    # w_dyde = np.gradient(angular_correction_matrix, ek_axis, axis=1)
    # w_dyda = np.gradient(angular_correction_matrix, angle_axis, axis=0)
    # w_dxda = 0
    # w_dxde = np.gradient(e_correction, ek_axis, axis=0)

    jacobian_determinant = calculate_jacobian(
        angular_correction_matrix,
        e_correction,
        ek_axis,
        angle_axis,
    )

    # attempt to update the dictionary
    # config_dict[""].update(elem)

    #  ek_axis = self._correction_matrix_dict[lens_mode][pass_energy][
    #             kinetic_energy
    #             ]["ek_axis"]
    #         angle_axis = self._correction_matrix_dict[lens_mode][pass_energy][
    #            kinetic_energy
    #         ]["angle_axis"]
    return (
        ek_axis,
        angle_axis,
        angular_correction_matrix,
        e_correction,
        jacobian_determinant,
    )


def calculate_jacobian(
    angular_correction_matrix,
    e_correction,
    ek_axis,
    angle_axis,
):
    w_dyde = np.gradient(angular_correction_matrix, ek_axis, axis=1)
    w_dyda = np.gradient(angular_correction_matrix, angle_axis, axis=0)
    w_dxda = 0
    w_dxde = np.gradient(e_correction, ek_axis, axis=0)
    jacobian_determinant = np.abs(w_dxde * w_dyda - w_dyde * w_dxda)
    return jacobian_determinant


# original function using regular grid interpolator
def physical_unit_data_1(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (
        angular_correction_matrix.flatten(),
        e_correction_expand.flatten(),
    )
    # these coords seems to be pixels..

    # x_bins = np.arange(0, image.shape[0], 1)
    # y_bins = np.arange(0, image.shape[1], 1)

    x_bins = np.arange(0, image.shape[0], 1)
    y_bins = np.arange(0, image.shape[1], 1)

    # print("x_bins-shape",  x_bins.shape)
    # print("y_bins-shape",  y_bins.shape)
    # print("image-shape",  image.shape)
    # create interpolation function
    angular_interpolation_function = RegularGridInterpolator(
        (x_bins, y_bins),
        image,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    corrected_data = (
        np.reshape(
            angular_interpolation_function(coords),
            angular_correction_matrix.shape,
        )
        * jacobian_determinant
    )

    return corrected_data


# fails due to memory allocation
# interpolate.interp2d
# Loop version
def physical_unit_data_2(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (angular_correction_matrix, e_correction_expand)
    # these coords seems to be pixels..
    x_bins = np.arange(0, image.shape[1], 1)
    y_bins = np.arange(0, image.shape[0], 1)

    # create interpolation function
    angular_interpolation_function = interpolate.interp2d(
        x_bins,
        y_bins,
        image,
        kind="linear",
        # bounds_error=False,
        # fill_value=0
    )

    # Loop, might be slow
    corrected_data = np.zeros(angular_correction_matrix.shape)
    for i in range(0, angular_correction_matrix.shape[0]):
        for j in range(0, angular_correction_matrix.shape[1]):
            corrected_data[i][j] = (
                angular_interpolation_function(
                    # e_correction,
                    # angular_correction_matrix[:, 0]
                    e_correction[j],
                    angular_correction_matrix[i][j],
                )
                * jacobian_determinant[i][j]
            )

    return corrected_data


# Using xarray internal interpolation functions
def physical_unit_data_3(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
    ek_axis,
    angle_axis,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (
        angular_correction_matrix.flatten(),
        e_correction_expand.flatten(),
    )
    # these coords seems to be pixels..

    x_bins = np.arange(0, image.shape[0], 1)
    y_bins = np.arange(0, image.shape[1], 1)

    # print("x_bins-shape",  x_bins.shape)
    # print("y_bins-shape",  y_bins.shape)
    # print("image-shape",  image.shape)

    # make the xarray
    image_xr = xr.DataArray(
        image,
        dims=["x", "y"],
        coords={"x": x_bins, "y": y_bins},
    )

    angle_xr = xr.DataArray(
        angular_correction_matrix,
        dims=["angle", "energy"],
        coords={"energy": ek_axis, "angle": angle_axis},
    )

    energy_xr = xr.DataArray(
        e_correction,
        dims=["energy"],
        coords={"energy": ek_axis},
    )

    corrected_data = (
        image_xr.interp(x=angle_xr, y=energy_xr).to_numpy()
        * jacobian_determinant
    )

    return corrected_data


# numba accelerated bilinear interpolation..
# https://stackoverflow.com/questions/8661537/
# how-to-perform-bilinear-interpolation-in-python
@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bilinear_interpolation(x_in, y_in, f_in, x_out, y_out):
    f_out = np.zeros((y_out.size, x_out.size))

    for i in prange(f_out.shape[1]):
        idx = np.searchsorted(x_in, x_out[i])

        x1 = x_in[idx - 1]
        x2 = x_in[idx]
        x = x_out[i]

        for j in prange(f_out.shape[0]):
            idy = np.searchsorted(y_in, y_out[j])
            y1 = y_in[idy - 1]
            y2 = y_in[idy]
            y = y_out[j]

            f11 = f_in[idy - 1, idx - 1]
            f21 = f_in[idy - 1, idx]
            f12 = f_in[idy, idx - 1]
            f22 = f_in[idy, idx]

            f_out[j, i] = (
                f11 * (x2 - x) * (y2 - y)
                + f21 * (x - x1) * (y2 - y)
                + f12 * (x2 - x) * (y - y1)
                + f22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))

    return f_out


# numba bilinear interpolation
def physical_unit_data_4(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
    ek_axis,
    angle_axis,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (
        angular_correction_matrix.flatten(),
        e_correction_expand.flatten(),
    )
    # these coords seems to be pixels..

    x_bins = np.arange(0, image.shape[0], 1)
    y_bins = np.arange(0, image.shape[1], 1)

    print("x_bins-shape", x_bins.shape)
    print("y_bins-shape", y_bins.shape)
    print("image-shape", image.shape)

    corrected_data = (
        bilinear_interpolation(
            x_bins,
            y_bins,
            image,
            e_correction_expand.flatten(),
            angular_correction_matrix.flatten(),
        ).reshape(angular_correction_matrix.shape)
        * jacobian_determinant
    )

    return corrected_data


# this seems to fail to memory allocation problems


# map_coordinates function
def physical_unit_data_5(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
    ek_axis,
    angle_axis,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = np.array(
        (angular_correction_matrix.flatten(), e_correction_expand.flatten()),
    )
    # print(coords.shape)
    # these coords seems to be pixels..

    x_bins = np.arange(0, image.shape[0], 1)
    y_bins = np.arange(0, image.shape[1], 1)

    # print("x_bins-shape",  x_bins.shape)
    # print("y_bins-shape",  y_bins.shape)
    # print("image-shape",  image.shape)

    corrected_data = (
        map_coordinates(image, coords, order=1).reshape(
            angular_correction_matrix.shape,
        )
        * jacobian_determinant
    )

    return corrected_data


# interpn, wrapper for regular grid interpolator simailar to original version
def physical_unit_data_6(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
    ek_axis,
    angle_axis,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (angular_correction_matrix, e_correction_expand)

    # these coords seems to be pixels..

    x_bins = np.arange(0, image.shape[0], 1)
    y_bins = np.arange(0, image.shape[1], 1)

    # print("x_bins-shape",  x_bins.shape)
    # print("y_bins-shape",  y_bins.shape)
    # print("image-shape",  image.shape)

    points = (x_bins, y_bins)

    corrected_data = (
        interpn(
            points,
            image,
            coords,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        * jacobian_determinant
    )

    return corrected_data


# Using the interpolation package
# in theory, numba implementation
# https://www.econforge.org/interpolation.py/
def physical_unit_data_7(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
    ek_axis,
    angle_axis,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    x_bins = np.arange(0, image.shape[0], 1).astype("float")
    y_bins = np.arange(0, image.shape[1], 1).astype("float")
    image_x_pts = image.shape[0]
    image_y_pts = image.shape[1]

    # uniform cartesian grid
    grid = UCGrid(
        (np.min(x_bins), np.max(x_bins), image_x_pts),
        (np.min(y_bins), np.max(y_bins), image_y_pts),
    )
    # get grid points
    # gp = nodes(grid)

    # coordinates for evaluation
    coords = np.array(
        (
            np.transpose(angular_correction_matrix.flatten()),
            np.transpose(e_correction_expand.flatten()),
        ),
    )
    # Modify for coords for the eval_linear function
    coords = np.transpose(coords)
    coords = np.array(coords, order="C")
    # print("numba type of coords = ", nb.typeof(coords))

    # pre-allocate result
    corrected_data = np.zeros(angular_correction_matrix.shape)

    # print("x_bins-shape",  x_bins.shape)
    # print("y_bins-shape",  y_bins.shape)
    # print("image-shape",  image.shape)
    # print("UCgrid ", grid)

    corrected_data = (
        eval_linear(grid, image, coords).reshape(
            angular_correction_matrix.shape,
        )
        * jacobian_determinant
    )

    return corrected_data


# LAURENZ SUGGESTIONS
# scipy.interpolate.RectBivariateSpline
# Loop version
def physical_unit_data_8(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates
    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (angular_correction_matrix, e_correction_expand)
    # these coords seems to be pixels..

    x_bins = np.arange(0, image.shape[1], 1)
    y_bins = np.arange(0, image.shape[0], 1)

    # create interpolation function

    angular_interpolation_function = interpolate.RectBivariateSpline(
        x_bins,
        y_bins,
        image.T,
        # kind='linear'
        kx=1,
        ky=1,
        # bounds_error=False,
        # fill_value=0
    )

    # Loop version, might be slow
    corrected_data = np.zeros(angular_correction_matrix.shape)
    for i in range(0, angular_correction_matrix.shape[0]):
        for j in range(0, angular_correction_matrix.shape[1]):
            corrected_data[i][j] = (
                angular_interpolation_function(
                    # e_correction,
                    # angular_correction_matrix[:, 0]
                    e_correction[j],
                    angular_correction_matrix[i][j],
                )
                * jacobian_determinant[i][j]
            )

    return corrected_data


# LAURENZ SUGGESTIONS
# scipy.interpolate.griddata
# this seems a wrapper for the NearestNDInterpolator,
# LinearNDInterpolator ,
# CloughTocher2DInterpolator functions
# depending on the method paramter
# this cannot work as the coordinates where we want to interpolate are not an uniform grid..
# let's try directly the linearND
# this also does not seem to work on our regualry spaced grid..seems to give problems with the
# triangulatio function, the example on the scipy uses a rando x and y distribution of input points..
def physical_unit_data_9(
    image,
    angular_correction_matrix,
    e_correction,
    jacobian_determinant,
):
    """_summary_

    Args:
        raw_data (_type_): _description_
        angular_correction_matrix (_type_): _description_
        e_correction (_type_): _description_
        jacobian_determinant (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create 2d matrix with the
    # ek coordinates

    e_correction_expand = (
        np.ones(angular_correction_matrix.shape) * e_correction
    )

    # x_bins = np.arange(0, image.shape[0], 1)
    # y_bins = np.arange(0, image.shape[1], 1)

    x_bins = np.arange(0, image.shape[0], 1)
    y_bins = np.arange(0, image.shape[1], 1)

    print("x_bins-shape", x_bins.shape)
    print("y_bins-shape", y_bins.shape)
    print("image-shape", image.shape)
    # create interpolation function

    meshcoords = list(zip(x_bins, y_bins))

    angular_interpolation_function = LinearNDInterpolator(meshcoords, image)

    # define new coordinates
    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    x_bins_new = (angular_correction_matrix.flatten(),)
    y_bins_new = e_correction_expand.flatten()

    corrected_data = angular_interpolation_function(
        angular_correction_matrix,
        e_correction_expand,
    )

    # corrected_data = (
    #     np.reshape(
    #         angular_interpolation_function(e_correction_expand,angular_correction_matrix),
    #         angular_correction_matrix.shape,
    #     ) *
    #     jacobian_determinant
    # )

    """ (
        np.reshape(
            angular_interpolation_function(x_bins_new,y_bins_new),
            angular_correction_matrix.shape,
        ) *
        jacobian_determinant
    )
 """
    return corrected_data


# scipy.interpolate.LinearNDInterpolator
