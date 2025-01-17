"""Specsanalyzer image conversion module"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import map_coordinates

# Configure logging
logger = logging.getLogger("specsanalyzer.specsscan")


def get_damatrix_from_calib2d(
    lens_mode: str,
    kinetic_energy: float,
    pass_energy: float,
    work_function: float,
    calib2d_dict: dict,
) -> tuple[float, np.ndarray, float, str, list[str]]:
    """This function estimates the best angular conversion coefficients for the current analyzer
    mode, starting from a dictionary containing the specs .calib2d database. A linear interpolation
    is performed from the tabulated coefficients based on the retardation ratio value.

    Args:
        lens_mode (str): the lens mode string description
        kinetic_energy (float): kinetic energy of the photoelectron
        pass_energy (float): analyzer pass energy
        work_function (float): work function settings
        calib2d_dict (dict): dictionary containing the configuration parameters for angular
            correction

    Returns:
        tuple[float, np.ndarray, float, str, list[str]]: (a_inner, da_matrix, retardation_ratio,
        source, dims) interpolated a_inner and da_matrix, needed for the coordinate conversion,
        retardation ratio, interpolation values, dims
    """

    # retardation ratio
    retardation_ratio = (kinetic_energy - work_function) / pass_energy

    # check the angular mode type
    try:
        supported_angle_modes = calib2d_dict["supported_angle_modes"]
        supported_space_modes = calib2d_dict["supported_space_modes"]
    except KeyError as exc:
        raise KeyError(
            "The supported modes were not found in the calib2d dictionary",
        ) from exc

    if lens_mode in supported_angle_modes:
        # given the lens mode get all the retardation ratios available
        rr_vec, da_matrix_full = get_rr_da(lens_mode, calib2d_dict)
        # find closest retardation ratio in table
        closest_rr_index = bisection(rr_vec, retardation_ratio)

        # return as the closest rr index the smallest index in
        # case of -1 output from bisection

        if closest_rr_index == -1:
            closest_rr_index = 0
        # now compare the distance with the neighboring indexes,
        # we need the second nearest rr
        second_closest_rr_index = second_closest_rr(rr_vec, closest_rr_index)

        # compute the rr_factor, which in igor done by a BinarySearchInterp
        # this is a fraction from the current rr to next rr in the table
        # array of array indexes
        rr_index = np.arange(0, rr_vec.shape[0], 1)
        # the factor is obtained by linear interpolation
        rr_factor = np.interp(retardation_ratio, rr_vec, rr_index) - closest_rr_index

        source = (
            f"interpolated as {(1-rr_factor)}*{lens_mode}@{rr_vec[closest_rr_index]}"
            f" + {rr_factor}*{lens_mode}@{rr_vec[second_closest_rr_index]}"
        )

        # weighted average between two neighboring da matrices
        da_matrix = (
            da_matrix_full[closest_rr_index][:][:] * (1 - rr_factor)
            + da_matrix_full[second_closest_rr_index][:][:] * rr_factor
        )
        # separate the first line (aInner) from the da coefficients
        a_inner = da_matrix[0][0]
        da_matrix = da_matrix[1:][:]

        dims = ["Angle", "Ekin"]

    elif lens_mode in supported_space_modes:
        # use the mode defaults
        logger.info("This is a spatial mode, using default " + lens_mode + " config")
        rr_vec, da_matrix_full = get_rr_da(lens_mode, calib2d_dict)
        a_inner = da_matrix_full[0][0]
        da_matrix = da_matrix_full[1:][:]
        source = f"{lens_mode}@default"

        dims = ["Position", "Ekin"]

    else:
        raise ValueError(f"Unrecognized lens mode '{lens_mode}'")

    return a_inner, da_matrix, retardation_ratio, source, dims


def bisection(array: np.ndarray, value: float) -> int:
    """
    Auxiliary function to find the closest rr index from https://stackoverflow.com/questions/2566412/
    find-nearest-value-in-numpy-array

    Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between
    array[j] and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is
    returned to indicate that ``value`` is out of range below and above respectively.
    This should mimic the function BinarySearch in igor pro 6

    Args:
        array (np.ndarray): ordered array
        value (float): comparison value

    Returns:
        int: index (non-integer) expressing the position of value between array[j] and array[j+1]
    """

    num_elems = len(array)
    if value < array[0]:
        return -1
    if value > array[num_elems - 1]:
        return num_elems
    lower_index = 0  # Initialize lower
    upper_index = num_elems - 1  # and upper limits.
    while upper_index - lower_index > 1:  # If we are not yet done,
        # compute a midpoint with a bitshift
        middle_index = (upper_index + lower_index) >> 1
        if value >= array[middle_index]:
            lower_index = middle_index  # and replace either the lower limit
        else:
            upper_index = middle_index  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    if value == array[num_elems - 1]:  # and top
        return num_elems - 1

    return lower_index


def second_closest_rr(rrvec: np.ndarray, closest_rr_index: int) -> int:
    """Return closest_rr_index+1 unless you are at the edge of the rrvec.

    Args:
        rrvec (np.ndarray): the retardation ratio vector
        closest_rr_index (int): the nearest rr index corresponding to the scan

    Returns:
        int: nearest rr index to calculate the best da coefficients
    """
    if closest_rr_index == (rrvec.size - 1):
        # we are the edge: the behavior is to not change the index
        second_closest_rr_index = closest_rr_index
    else:
        second_closest_rr_index = closest_rr_index + 1

    return second_closest_rr_index


def get_rr_da(
    lens_mode: str,
    calib2d_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the retardation ratios and the da for a certain lens mode from the configuration
    dictionary

    Args:
        lens_mode (string): string containing the lens mode
        calib2d_dict (dict): dictionary containing the configuration parameters for angular
            correction

    Raises:
        KeyError: Raised if the requested lens mode is not found
        ValueError: Raised if no da values are found for the given mode

    Returns:
        tuple[np.ndarray, np.ndarray]: rr vector, matrix of da coefficients
        per row row0 : da1, row1: da3, .. up to da7.
        Non angle resolved lens modes do only posses da1.
    """
    # check if this is spatial or an angular mode
    # check the angular mode type
    try:
        supported_angle_modes = calib2d_dict["supported_angle_modes"]
        supported_space_modes = calib2d_dict["supported_space_modes"]
    except KeyError as exc:
        raise KeyError(
            "The supported modes were not found in the calib2d dictionary",
        ) from exc

    if lens_mode in supported_angle_modes:
        rr_array = np.array(list(calib2d_dict[lens_mode]["rr"]))

        dim1 = rr_array.shape[0]
        base_dict = calib2d_dict[lens_mode]["rr"]
        dim2 = len(base_dict[rr_array[0]])

        try:
            dim3 = len(base_dict[rr_array[0]]["Da1"])
        except KeyError as exc:
            raise ValueError(
                "Da values do not exist for the given mode.",
            ) from exc

        da_matrix = np.zeros([dim1, dim2, dim3])
        for count, item in enumerate(rr_array):
            a_inner = base_dict[item]["aInner"]
            da_block = np.concatenate(
                tuple([v] for k, v in base_dict[item].items() if k != "aInner"),
            )
            da_matrix[count] = np.concatenate(
                (np.array([[a_inner] * dim3]), da_block),
            )

    elif lens_mode in supported_space_modes:
        # ok we are in a spatial mode, the calib2d does
        # not contain an rr_array and we should build the da_matrix from the
        # defaults without interpolation

        base_dict = calib2d_dict[lens_mode]["default"]
        da1 = np.array(base_dict["Da1"])
        a_inner = base_dict["aInner"]
        rr_array = np.ones(1)
        da_matrix = np.zeros((4, 3))
        da_matrix[0, :] = np.ones(3) * a_inner
        da_matrix[1, :] = da1

    else:
        raise ValueError(f"Unrecognized lens mode '{lens_mode}")

    return rr_array, da_matrix


def calculate_polynomial_coef_da(
    da_matrix: np.ndarray,
    kinetic_energy: float,
    pass_energy: float,
    e_shift: np.ndarray,
) -> np.ndarray:
    """Given the da coefficients contained in the scan parameters, the program calculates the energy
    range based on the eshift parameter and fits a second order polynomial to the tabulated values.
    The polynomial coefficients are packed in the dapolymatrix array (row0 da1, row1 da3, ..)
    The function returns a matrix of the fit coefficients, given the physical energy scale
    Each line of the matrix is a set of coefficients for each of the da[i] corrections

    Args:
        da_matrix (np.ndarray): the matrix of interpolated da coefficients
        kinetic_energy (float): photoelectron kinetic energy
        pass_energy (float): analyzer pass energy
        e_shift (np.ndarray): e shift parameter, defining the energy
            range around the center for the polynomial fit of the da coefficients

    Returns:
        np.ndarray: dapolymatrix containing the fit results (row0 da1, row1 da3, ..)
    """
    # calculate the energy values for each da, given the eshift
    da_energy = e_shift * pass_energy + kinetic_energy * np.ones(e_shift.shape)

    # create the polynomial coefficient matrix, each is a second order polynomial
    da_poly_matrix = np.zeros(da_matrix.shape)

    for i in range(0, da_matrix.shape[0]):
        # igor uses the fit poly 3, which should be a parabola
        da_poly_matrix[i][:] = np.polyfit(
            da_energy,
            da_matrix[i][:],
            2,
        ).transpose()

    return da_poly_matrix


def zinner(
    kinetic_energy: np.ndarray,
    angle: np.ndarray,
    da_poly_matrix: np.ndarray,
) -> np.ndarray:
    """Auxiliary function for mcp_position_mm, uses kinetic energy and angle starting from the
    dapolymatrix, to get the zinner coefficient to calculate the electron arrival position on the
    mcp withing the a_inner boundaries

    Args:
        kinetic_energy (np.ndarray): kinetic energies
        angle (np.ndarray): angles
        da_poly_matrix (np.ndarray): matrix with polynomial coefficients

    Returns:
        np.ndarray: returns the calculated positions on the mcp, valid for low angles  (< a_inner)
    """
    out = np.zeros(angle.shape, float)

    for i in np.arange(0, da_poly_matrix.shape[0], 1):
        out = out + (
            (10.0 ** (-2 * i))
            * (angle ** (1 + 2 * i))
            * np.polyval(da_poly_matrix[i][:], kinetic_energy)
        )
    return out


def zinner_diff(
    kinetic_energy: np.ndarray,
    angle: np.ndarray,
    da_poly_matrix: np.ndarray,
) -> np.ndarray:
    """Auxiliary function for mcp_position_mm, uses kinetic energy and angle starting from the
    dapolymatrix, to get the zinner_diff coefficient to correct the electron arrival position on
    the mcp outside the a_inner boundaries

    Args:
        kinetic_energy (np.ndarray): kinetic energies
        angle (np.ndarray): angles
        da_poly_matrix (np.ndarray): polynomial matrix

    Returns:
        np.ndarray: zinner_diff the correction for the zinner position on the MCP for high
        (>a_inner) angles.
    """

    out = np.zeros(angle.shape, float)

    for i in np.arange(0, da_poly_matrix.shape[0], 1):
        out = out + (
            (10.0 ** (-2 * i))
            * (1 + 2 * i)
            * (angle ** (2 * i))
            * np.polyval(da_poly_matrix[i][:], kinetic_energy)
        )

    return out


def mcp_position_mm(
    kinetic_energy: np.ndarray,
    angle: np.ndarray,
    a_inner: float,
    da_poly_matrix: np.ndarray,
) -> np.ndarray:
    """calculated the position of the photoelectron on the mcp, for a certain kinetic energy and
    emission angle. This is determined for the given lens mode (as defined by the a_inner and
    dapolymatrix)

    Args:
        kinetic_energy (np.ndarray): kinetic energies
        angle (np.ndarray): photoemission angles
        a_inner (float): inner angle parameter of the lens mode
        da_poly_matrix (np.ndarray): matrix with the polynomial correction coefficients for
            calculating the arrival position on the MCP
    Returns:
        np.ndarray: lateral position of photoelectron on the mcp (angular dispersing axis)
    """

    # define two angular regions: within and outside the a_inner boundaries
    mask = np.less_equal(np.abs(angle), a_inner)

    a_inner_vec = np.ones(angle.shape) * a_inner

    result = np.where(
        mask,
        zinner(kinetic_energy, angle, da_poly_matrix),
        np.sign(angle)
        * (
            zinner(kinetic_energy, a_inner_vec, da_poly_matrix)
            + (np.abs(angle) - a_inner_vec)
            * zinner_diff(kinetic_energy, a_inner_vec, da_poly_matrix)
        ),
    )
    return result


def calculate_matrix_correction(
    kinetic_energy: float,
    pass_energy: float,
    nx_pixels: int,
    ny_pixels: int,
    pixel_size: float,
    magnification: float,
    e_shift: np.ndarray,
    de1: float,
    e_range: np.ndarray,
    a_range: np.ndarray,
    a_inner: float,
    da_matrix: np.ndarray,
    angle_offset_px: int,
    energy_offset_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the angular and energy interpolation matrices for the correction function.

    Args:
        kinetic_energy (float): analyzer set kinetic energy
        pass_energy (float): analyzer set pass energy
        nx_pixels (int): number of image pixels (after binning) along the energy dispersing
            direction
        ny_pixels (int): number of image pixels (after binning) along the angle/spatially
            dispersing direction
        pixel_size (float): pixel size in millimeter
        magnification (float): magnification of the lens system used for imaging the detector
        e_shift (np.ndarray): e shift parameter, defining the energy
            range around the center for the polynomial fit of the da coefficients
        de1 (float): energy dispersion factor (fraction of pass_energy)/mm_z)
        e_range (np.ndarray): energy range (minimal/maximal energy, in units of pass_energy)
        a_range (np.ndarray): angular/spatial range (minimal/maximal angle or distance, in deg
            or mm)
        a_inner (float): inner angle parameter of the lens mode
        da_matrix (np.ndarray): the matrix of interpolated da coefficients
        angle_offset_px (int): Angular offset in pixel
        energy_offset_px (int): Energy offset in pixel

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - ek_axis: kinetic energy axis
            - angle_axis, angle of emission axis
            - angular_correction_matrix: the matrix for angular interpolation
            - e_correction: the matrix for energy interpolation
            - jacobian_determinant: the transformation jacobian for area preserving transformation
    """
    da_poly_matrix = calculate_polynomial_coef_da(da_matrix, kinetic_energy, pass_energy, e_shift)

    ek_low = kinetic_energy + e_range[0] * pass_energy
    ek_high = kinetic_energy + e_range[1] * pass_energy

    ek_axis = np.linspace(ek_low, ek_high, nx_pixels, endpoint=False)
    angle_low = a_range[0] * 1.2
    angle_high = a_range[1] * 1.2

    angle_axis = np.linspace(angle_low, angle_high, ny_pixels, endpoint=False)

    mcp_position_mm_matrix = np.zeros([nx_pixels, ny_pixels], dtype=float)
    angular_correction_matrix = np.zeros([nx_pixels, ny_pixels], dtype=float)
    e_correction = np.zeros(ek_axis.shape)

    # create a meshgrid for vectorized evaluation
    ek_mesh, angle_mesh = np.meshgrid(ek_axis, angle_axis)

    mcp_position_mm_matrix = mcp_position_mm(ek_mesh, angle_mesh, a_inner, da_poly_matrix)

    angular_correction_matrix = (
        mcp_position_mm_matrix / magnification / pixel_size + ny_pixels / 2 + angle_offset_px
    )

    e_correction = (
        (ek_axis - kinetic_energy * np.ones(ek_axis.shape))
        / pass_energy
        / de1
        / magnification
        / pixel_size
        + nx_pixels / 2
        + energy_offset_px
    )

    # calculate the Jacobian determinant of the transformation
    jacobian_determinant = calculate_jacobian(
        angular_correction_matrix,
        e_correction,
        ek_axis,
        angle_axis,
    )

    return ek_axis, angle_axis, angular_correction_matrix, e_correction, jacobian_determinant


def calculate_jacobian(
    angular_correction_matrix: np.ndarray,
    e_correction: np.ndarray,
    ek_axis: np.ndarray,
    angle_axis: np.ndarray,
) -> np.ndarray:
    """calculate the jacobian matrix associated with the transformation

    Args:
        angular_correction_matrix (np.ndarray): angular correction matrix
        e_correction (np.ndarray): energy correction
        ek_axis (np.ndarray): kinetic energy axis
        angle_axis (np.ndarray): angle axis

    Returns:
        np.ndarray: jacobian_determinant matrix
    """
    w_dyde = np.gradient(angular_correction_matrix, ek_axis, axis=1)
    w_dyda = np.gradient(angular_correction_matrix, angle_axis, axis=0)
    w_dxda = 0
    w_dxde = np.gradient(e_correction, ek_axis, axis=0)
    jacobian_determinant = np.abs(w_dxde * w_dyda - w_dyde * w_dxda)
    return jacobian_determinant


def physical_unit_data(
    image: np.ndarray,
    angular_correction_matrix: np.ndarray,
    e_correction: float,
    jacobian_determinant: np.ndarray,
) -> np.ndarray:
    """interpolate the image on physical units, using the ``map_coordinates`` function from
    ``scipy.ndimage``

    Args:
        image (np.ndarray): raw image
        angular_correction_matrix (np.ndarray): angular correction matrix
        e_correction (float): energy correction
        jacobian_determinant (np.ndarray): jacobian determinant for preserving
            the area normalization

    Returns:
        np.ndarray: interpolated image as a function of angle and energy
    """

    # Create a list of e and angle pixel
    # coordinates where to
    # evaluate the interpolating
    # function

    # create a 2d matrix with the
    # y pixel coordinates for a certain kinetic energy
    e_correction_matrix = np.ones(angular_correction_matrix.shape) * e_correction

    # flatten the x and y to a 2 x N coordinates array
    # N = Nxpixels x Nypixels
    coords = np.array([angular_correction_matrix.flatten(), e_correction_matrix.flatten()])

    # the image is expressed as intensity vs pixels,
    # angular correction and e_correction
    corrected_data = (
        map_coordinates(image, coords, order=1).reshape(angular_correction_matrix.shape)
        * jacobian_determinant
    )

    return corrected_data
