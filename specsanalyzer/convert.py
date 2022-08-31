import numpy as np
from scipy.ndimage import map_coordinates


def get_damatrix_fromcalib2d(
    lens_mode: str,
    kinetic_energy: float,
    pass_energy: float,
    work_function: float,
    config_dict: dict,
) -> tuple[float, np.ndarray]:
    """This function estimates the best angular
    conversion coefficients for the current analyser mode, starting from
    a dictionary containing the specs .calib2d database.
    A linear interpolation is performed from the tabulated coefficients based
    on the retardatio ratio value.

    Args:
        lens_mode (string): the lens mode string description
        kinetic_energy (float): _description_
        pass_energy (float): _description_
        work_function (float): _description_
        config_dict (dict): _description_

    Returns:
        Tuple[float,np.ndarray]: _description_
    """

    # retardation ratio
    retardation_ratio = (kinetic_energy - work_function) / pass_energy

    # given the lens mode get all the retardation ratios available
    rr_vec, damatrix_full = get_rr_da(lens_mode, config_dict)
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

    damatrix_close = damatrix_full[closest_rr_index][:][:]
    damatrix_second = damatrix_full[second_closest_rr_index][:][:]
    one_mat = np.ones(damatrix_close.shape)
    rr_factor_mat = np.ones(damatrix_close.shape) * rr_factor
    # weighted average between two neighboring da matrices
    damatrix = (
        damatrix_close * (one_mat - rr_factor_mat) + damatrix_second * rr_factor_mat
    )
    # separate the first line (aInner) from the da coefficients
    a_inner = damatrix[0][0]
    damatrix = damatrix[1:][:]

    return a_inner, damatrix


# Auxiliary function to find the closest rr index
# from https://stackoverflow.com/questions/2566412/
# find-nearest-value-in-numpy-array


def bisection(array: np.ndarray, value: float) -> int:
    """ "" Given an ``array`` , and given a ``value`` , returns an index
    j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic
    increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    This should mimick the function BinarySearch in igor pro 6

    Args:
        array (np.ndarray): ordered array
        value (float): comparison value

    Returns:
        int: index (non-integer) expressing the position of value between
        array[j] and array[j+1]
    """

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


def second_closest_rr(rrvec: np.ndarray, closest_rr_index: int) -> int:
    """Return closest_rr_index+1 unless you are at the edge
    of the rrvec.

    Args:
        rrvec (np.ndarray): the retardation ratio vector
        closest_rr_index (int): the nearest rr index corresponding to the scan

    Returns:
        int: nearest rr index to calculate the best da coefficients
    """
    if closest_rr_index == (rrvec.size - 1):
        # we are the edge: the behaviour is to not change the index
        second_closest_rr_index = closest_rr_index
    else:
        second_closest_rr_index = closest_rr_index + 1

    return second_closest_rr_index


def get_rr_da(
    lens_mode: str,
    config_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the retardatio ratios and the da for a certain lens mode from the
    confugaration dictionary

    Args:
        lens_mode (string): string containing the lens mode
        config_dict (dict): config dictionary

    Raises:
        Da1Error: to be tested

    Returns:
        Tuple[np.ndarray,np.ndarray]: retardation ratio vector, matrix of
        da coeffients, per row row0 : da1, row1: da3, .. up to da7
        non angle resolved lens modes do not posses das.

    """
    rr_array = np.array(list(config_dict["calib2d_dict"][lens_mode]["rr"]))

    dim1 = rr_array.shape[0]
    base_dict = config_dict["calib2d_dict"][lens_mode]["rr"]
    dim2 = len(base_dict[rr_array[0]])

    try:
        dim3 = len(base_dict[rr_array[0]]["Da1"])
    except KeyError as exc:
        raise Da1Error() from exc

    da_matrix = np.zeros([dim1, dim2, dim3])
    for count, item in enumerate(rr_array):
        a_inner = base_dict[item]["aInner"]
        da_block = np.concatenate(
            tuple([v] for k, v in base_dict[item].items() if k != "aInner"),
        )
        da_matrix[count] = np.concatenate((np.array([[a_inner] * dim3]), da_block))
    return rr_array, da_matrix


def calculate_polynomial_coef_da(
    da_matrix: np.ndarray,
    kinetic_energy: float,
    pass_energy: float,
    eshift: float,
) -> np.ndarray:
    """Given the da coeffiecients contained in the
    scanpareters, the program calculate the energy range based
    on the eshift parameter and fits a second order polinomial
    to the tabulated values. The polinomial coefficients
    are packed in the dapolymatrix array (row0 da1, row1 da3, ..)
    The dapolymatrix is also saved in the scanparameters dictionary

    Args:
        da_matrix (np.ndarray): _description_
        kinetic_energy (float): _description_
        pass_energy (float): _description_
        eshift (float): _description_

    Returns:
        np.ndarray: _description_
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


def zinner(
    ek: float,
    angle: float,
    dapolymatrix: np.ndarray,
) -> float:
    """Auxiliary function for mcp_position_mm, uses kinetic energy and angle
    starting from the dapolymatrix, to get
    the zinner coefficient to calculate the electron arrival position on the
    mcp withing the a_inner boundaries

    Args:
        ek (float): kinetic energy
        angle (float): angle
        dapolymatrix (np.ndarray): matrix with polynomial coefficients

    Returns:
        float: zinner,
    """
    out = 0

    for i in np.arange(0, dapolymatrix.shape[0], 1):
        out = out + (
            (10.0 ** (-2 * i))
            * (angle ** (1 + 2 * i))
            * np.polyval(dapolymatrix[i][:], ek)
        )
    return out


def zinner_diff(
    ek: float,
    angle: float,
    dapolymatrix: np.ndarray,
) -> float:
    """Auxiliary function for mcp_position_mm, uses kinetic energy and angle
    starting from the dapolymatrix, to get
    the zinner_diff coefficient to coorect the electron arrival position on the
    mcp outside the a_inner boundaries

    Args:
        ek (float): kinetic energy
        angle (float): angle
        dapolymatrix (np.ndarray): polinomial matrix

    Returns:
        float: zinner_diff
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


def mcp_position_mm(
    ek: float,
    angle: float,
    a_inner: float,
    dapolymatrix: np.ndarray,
) -> np.ndarray:
    """_summary_

    Args:
        ek (float): _description_
        angle (float): _description_
        a_inner (float): _description_
        dapolymatrix (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """

    # define two angular regions: within and outsied the a_inner boundaries
    mask = np.less_equal(np.abs(angle), a_inner)

    a_inner_vec = np.ones(angle.shape) * a_inner

    result = np.where(
        mask,
        zinner(ek, angle, dapolymatrix),
        np.sign(angle)
        * (
            zinner(ek, a_inner_vec, dapolymatrix)
            + (np.abs(angle) - a_inner_vec) * zinner_diff(ek, a_inner_vec, dapolymatrix)
        ),
    )
    return result


def calculate_matrix_correction(
    lens_mode: str,
    kinetic_energy: float,
    pass_energy: float,
    work_function: float,
    binning: int,
    config_dict: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the matrix correction function for the interpoolation

    Args:
        lens_mode (str): _description_
        kinetic_energy (float): _description_
        pass_energy (float): _description_
        work_function (float): _description_
        binning (int): _description_
        config_dict (dict): _description_

    Returns:
        tuple[np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray]: returns ek_axis,
        angle_axis,
        angular_correction_matrix,
        e_correction,
        jacobian_determinant,
    """
    eshift = np.array(config_dict["calib2d_dict"]["eShift"])

    aInner, damatrix = get_damatrix_fromcalib2d(
        lens_mode,
        kinetic_energy,
        pass_energy,
        work_function,
        config_dict,
    )

    dapolymatrix = calculate_polynomial_coef_da(
        damatrix,
        kinetic_energy,
        pass_energy,
        eshift,
    )

    de1 = [config_dict["calib2d_dict"]["De1"]]
    erange = config_dict["calib2d_dict"]["eRange"]
    arange = config_dict["calib2d_dict"][lens_mode]["default"]["aRange"]
    nx_pixel = config_dict["nx_pixel"]
    ny_pixel = config_dict["ny_pixel"]
    pixelsize = config_dict["pixel_size"]
    magnification = config_dict["magnification"]

    nx_bins = int(nx_pixel / binning)
    ny_bins = int(ny_pixel / binning)

    # the bins of the new image, defaul = the original image
    # get form the configuraton file an upsampling factor
    try:
        ke_upsampling_factor = config_dict["ke_upsampling_factor"]
        angle_upsampling_factor = config_dict["angle_upsampling_factor"]
    except KeyError:
        ke_upsampling_factor = 1
        angle_upsampling_factor = 1

    n_ke_bins = ke_upsampling_factor * nx_bins
    n_angle_bins = angle_upsampling_factor * ny_bins

    ek_low = kinetic_energy + erange[0] * pass_energy
    ek_high = kinetic_energy + erange[1] * pass_energy

    ek_axis = np.linspace(ek_low, ek_high, n_ke_bins, endpoint=False)
    angle_low = arange[0] * 1.2
    angle_high = arange[1] * 1.2

    angle_axis = np.linspace(
        angle_low,
        angle_high,
        n_angle_bins,
        endpoint=False,
    )

    mcp_position_mm_matrix = np.zeros([n_ke_bins, n_angle_bins])
    angular_correction_matrix = np.zeros([n_ke_bins, n_angle_bins])
    e_correction = np.zeros(ek_axis.shape)

    # let's create a meshgrid for vectorized evaluation
    ek_mesh, angle_mesh = np.meshgrid(ek_axis, angle_axis)

    mcp_position_mm_matrix = mcp_position_mm(
        ek_mesh,
        angle_mesh,
        aInner,
        dapolymatrix,
    )

    # read angular and energy offsets from configuration file
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

    # calculate the Jacobian determinant of the transformation
    jacobian_determinant = calculate_jacobian(
        angular_correction_matrix,
        e_correction,
        ek_axis,
        angle_axis,
    )

    return (
        ek_axis,
        angle_axis,
        angular_correction_matrix,
        e_correction,
        jacobian_determinant,
    )


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
        np.ndarray: _description_
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
    """interpolate the image on physical units, using the map_coordinates
    function from scipy.ndimage

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
    # N = Nxpix x Nypixels
    coords = np.array(
        (angular_correction_matrix.flatten(), e_correction_matrix.flatten()),
    )

    # the image is expressed as intensity vs pixels,
    # angular correction and e_correction
    corrected_data = (
        map_coordinates(image, coords, order=1).reshape(
            angular_correction_matrix.shape,
        )
        * jacobian_determinant
    )

    return corrected_data


# error class for the Da s


class Da1Error(KeyError):
    def __init__(self):
        super().__init__("Da values do not exist for the given mode.")
