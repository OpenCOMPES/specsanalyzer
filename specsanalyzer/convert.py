import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def convert_image(
    raw_image_name,
    infofilename,
    calib2dfilename,
):
    """_summary_

    Args:
        raw_image_name (_type_): _description_
        infofilename (_type_): _description_
        calib2dfilename (_type_): _description_

    Returns:
        _type_: _description_
    """

    raw_data = np.loadtxt(raw_image_name, delimiter="\t")
    get_damatrix_fromcalib2d(infofilename, calib2dfilename)

    scanparameters = get_scanparameters(infofilename, calib2dfilename)
    calculate_polynomial_coef_da(scanparameters)

    (
        ek_axis, angle_axis, angular_correction_matrix, e_correction,
        jacobian_determinant,
    ) = calculate_matrix_correction(scanparameters)

    corrected_data = physical_unit_data(
        raw_data,
        angular_correction_matrix,
        e_correction,
        jacobian_determinant,
    )

    return (
        ek_axis,
        angle_axis,
        corrected_data,
    )


def get_scanparameters(infofilename, calib2dfilename):
    """_summary_

    Args:
        infofilename (_type_): _description_
        calib2dfilename (_type_): _description_

    Returns:
        _type_: _description_
    """
    # start building the dictionary from the info.txt
    with open(infofilename) as fd:
        scanparameters = dict(get_pair(line) for line in fd)

    with open(calib2dfilename) as calibfile:
        calibfilelines = calibfile.readlines()
        # from the header of the calib2d, get
        # eShift = -0.05 0 0.05 # Ep
        # eRange = -0.066 0.066 # Ep
        # eGrid  = 0.01 # Ep
        # aGrid  = 1 # unit
        # De1    = 0.0030

        # find the header line
        head_string = "SPECS Phoibos2D"
        header_start_idx = [
            i for i, item in enumerate(calibfilelines)
            if re.search(head_string, item)
        ]

        for line_index in np.arange(header_start_idx[0]+1,
                                    header_start_idx[0]+13, 1):
            splitline = calibfilelines[line_index].split(" ")

            if splitline[0] == "eShift":
                scanparameters["eShift"] = [
                    float(splitline[2]),
                    float(splitline[3]),
                    float(splitline[4]),
                ]
            if splitline[0] == "eRange":
                scanparameters["eRange"] = [
                    float(splitline[2]),
                    float(splitline[3]),
                ]
            if splitline[0] == "eGrid":
                scanparameters["eGrid"] = [float(splitline[3])]
            if splitline[0] == "aGrid":
                scanparameters["aGrid"] = [float(splitline[3])]
            if splitline[0] == "De1":
                scanparameters["De1"] = [float(splitline[5][:-2])]

        # from the body of the calib2d, get:
        # these changes with the user selected parameters
        # aRange = -15 15

        mode_default_start_idx = [
            i for i, item in enumerate(calibfilelines)
            if (
                re.search("default", item)
                and re.search(scanparameters["LensMode"], item)
            )
        ]

        for line_index in np.arange(
            mode_default_start_idx[0]+1,
            mode_default_start_idx[0]+10, 1,
        ):

            splitline = calibfilelines[line_index].split(" ")
            if splitline[0] == "aRange":
                scanparameters["aRange"] = [
                    float(splitline[2]),
                    float(splitline[3]),
                ]

        # DEFINE THE DETECTOR PARAMETERS; currently hard-coded and not IO
        scanparameters['ny_pixel'] = 512*2
        scanparameters['nx_pixel'] = 688*2
        scanparameters['pixelsize'] = 0.00645
        scanparameters['magnification'] = 4.54
        scanparameters['wf'] = 4.2  # is this currently used?

    damatrix = get_damatrix_fromcalib2d(infofilename, calib2dfilename)
    scanparameters['aInner'] = damatrix[0][0]
    scanparameters['damatrix'] = damatrix[1:][:]

    return scanparameters


def get_damatrix_fromcalib2d(infofilename, calib2dfilename):
    """_summary_

    Args:
        infofilename (_type_): _description_
        calib2dfilename (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        infofile = open(infofilename)
        calibfile = open(calib2dfilename)

        # now read the infofile and return a dictionary
        infodict = dict(get_pair(line) for line in infofile)
        # now from the retardatio ratio and lens mode go through the calib file
        # and get the good stuff
        ek = float(infodict['KineticEnergy'])
        ep = float(infodict['PassEnergy'])
        lensmode = infodict['LensMode']
        rr = ek/ep

        # given the lens mode get all the retardatio ratios available
        calibfilelines = calibfile.readlines()
        rr_vec, damatrix_full = get_rr_da(calibfilelines, lensmode)
        closest_rr_index = bisection(rr_vec, rr)

        # return as the closest rr index the smallest in case of -1 output
        if closest_rr_index == -1:
            closest_rr_index = 0

        # now compare the distance with the neighboring indexes,
        # we need the second nearest rr
        second_closest_rr_index = second_closest_rr(rr, rr_vec,
                                                    closest_rr_index)

        # compute the rr_factor, in igor done by a BinarySearchInterp
        # find closest retardation ratio in table
        # rr_inf=BinarySearch(w_rr, rr)
        # fraction from this to next retardation ratio in table
        # rr_factor=BinarySearchInterp(w_rr, rr)-rr_inf
        rr_index = np.arange(0, rr_vec.shape[0], 1)
        rr_factor = np.interp(rr, rr_vec, rr_index)-closest_rr_index

        # print(rr_factor)

        damatrix_close = damatrix_full[closest_rr_index][:][:]
        damatrix_second = damatrix_full[second_closest_rr_index][:][:]
        # print(damatrix_close.shape)
        # print(damatrix_second.shape)

        one_mat = np.ones(damatrix_close.shape)
        rr_factor_mat = np.ones(damatrix_close.shape)*rr_factor
        damatrix = (
            damatrix_close*(one_mat-rr_factor_mat) +
            damatrix_second*rr_factor_mat
        )

        return damatrix
    except OSError:
        print("Error: File does not appear to exist.")
        return 0

# Auxiliary function to load the info file


def get_pair(line):
    """_summary_

    Args:
        line (_type_): _description_

    Returns:
        _type_: _description_
    """
    key, sep, value = line.strip().partition(":")
    return key, value

# Auxiliary function to find the closest rr index
# from https://stackoverflow.com/questions/2566412/
# find-nearest-value-in-numpy-array


def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index
    j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic
    increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0  # Initialize lower
    ju = n-1  # and upper limits.
    while (ju-jl > 1):  # If we are not yet done,
        jm = (ju+jl) >> 1  # compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):  # edge cases at bottom
        return 0
    elif (value == array[n-1]):  # and top
        return n-1
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
    if closest_rr_index == 0:
        second_closest_rr_index = 1
    else:
        if closest_rr_index == (rrvec.size-1):
            second_closest_rr_index = closest_rr_index-1
        else:
            # we are not at the edges, compare the neighbors and get the
            # closest
            if (rr < rrvec[closest_rr_index]):
                second_closest_rr_index = closest_rr_index-1
            else:
                second_closest_rr_index = closest_rr_index+1

    return second_closest_rr_index

# this function should get both the rr array, and the corresponding Da matrices
# for a certain Angle mode


def get_da_block(lines, blockstart, blocklenght):
    """_summary_

    Args:
        lines (_type_): _description_
        blockstart (_type_): _description_
        blocklenght (_type_): _description_

    Returns:
        _type_: _description_
    """
    damatrix = np.zeros([5, 3])
    for p, q in enumerate(range(blockstart+1, blockstart+1+blocklenght, 1)):

        linesplit = lines[q].split(" ")

        # print(linesplit)

        if linesplit[0] == "aInner":
            damatrix[p][:] = float(linesplit[2])
        else:
            for i, j in enumerate(range(2, 5, 1)):
                damatrix[p][i] = float(linesplit[j])
    return damatrix


def get_rr_da(lines, modestring):
    """_summary_

    Args:
        lines (_type_): _description_
        modestring (_type_): _description_

    Returns:
        _type_: _description_
    """
    rr = []

    if modestring == "WideAngleMode":
        lines_idx = [
            i for i, item in enumerate(lines)
            if (
                re.search(modestring, item) and
                not(re.search("Super", item))
            )
        ]
    else:
        lines_idx = [
            i for i, item in enumerate(lines)
            if re.search(modestring, item)
        ]

    block_start_list = []
    for i in lines_idx:
        wamline = lines[i]

        if (wamline.find("@") != -1):
            wamlinestrip = wamline[:-2].replace("]", "")
            wamlinestrip = wamlinestrip.replace("[", "")
            rr.append(float(wamlinestrip.split("@")[1]))
            block_start_list.append(i)

    # here we should have a get block function

    # maybe dangerous?
    if len(block_start_list) > 1:
        block_lenght = block_start_list[1]-block_start_list[0]-2
    else:
        block_lenght = 2

    rr_array = np.array(rr)
    # here we make a (rr lenght)x5x3 matrix
    da_matrix_full = np.zeros([rr_array.shape[0], 5, 3])

    for i, start_index in enumerate(block_start_list):

        block_da_matrix = get_da_block(lines, start_index, block_lenght)

        da_matrix_full[i][:][:] = block_da_matrix[:][:]

    return rr_array, da_matrix_full


def calculate_polynomial_coef_da(scanparameters):
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

    currentdamatrix = scanparameters['damatrix']
    eshift = np.array(scanparameters['eShift'])
    ek = float(scanparameters['KineticEnergy'])
    ep = float(scanparameters['PassEnergy'])

    # calcualte the energy values for each da, given the eshift
    da_energy = eshift*ep+ek*np.ones(eshift.shape)

    # create the polinomial coeffiecient matrix,
    # each is a third order polinomial

    dapolymatrix = np.zeros(currentdamatrix.shape)

    for i in range(0, currentdamatrix.shape[0]):
        # igor uses the fit poly 3, which should be a parabola
        dapolymatrix[i][:] = np.polyfit(
            da_energy,
            currentdamatrix[i][:], 2,
        ).transpose()

    scanparameters['dapolymatrix'] = dapolymatrix
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
        out = out + ((10.0**(-2*i)) *
                     (angle**(1+2*i)) *
                     np.polyval(dapolymatrix[i][:], ek))
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

        out = out + ((10.0**(-2*i)) *
                     (1+2*i) *
                     (angle**(2*i)) *
                     np.polyval(dapolymatrix[i][:], ek))

    return out


def mcp_position_mm(ek, angle, scanparameters):
    """_summary_

    Args:
        ek (_type_): _description_
        angle (_type_): _description_
        scanparameters (_type_): _description_

    Returns:
        _type_: _description_
    """

    ainner = scanparameters['aInner']
    dapolymatrix = scanparameters['dapolymatrix']

    mask = np.less_equal(np.abs(angle), ainner)

    # result=np.zeros(angle.shape)#ideally has to be evaluated on a mesh

    ainner_vec = np.ones(angle.shape)*ainner
    # result = np.where(mask,-10,10)
    result = np.where(
        mask, zinner(ek, angle, dapolymatrix),
        np.sign(angle)*(
            zinner(ek, ainner_vec, dapolymatrix) +
            (np.abs(angle)-ainner_vec) *
            zinner_diff(ek, ainner_vec, dapolymatrix)
        ),
    )
    return result


def calculate_matrix_correction(scanparameters):
    """_summary_

    Args:
        scanparameters (_type_): _description_

    Returns:
        _type_: _description_
    """

    ek = float(scanparameters["KineticEnergy"])
    ep = float(scanparameters["PassEnergy"])
    # dapolymatrix = scanparameters["dapolymatrix"]
    de1 = scanparameters["De1"]
    erange = scanparameters["eRange"]
    arange = scanparameters["aRange"]
    # ainner = scanparameters["aInner"]
    nx_pixel = scanparameters["nx_pixel"]
    ny_pixel = scanparameters["ny_pixel"]
    pixelsize = scanparameters["pixelsize"]
    binning = float(scanparameters["Binning"])*2
    magnification = scanparameters["magnification"]
    nx_bins = int(nx_pixel/binning)
    ny_bins = int(ny_pixel/binning)
    ek_low = ek + erange[0]*ep
    ek_high = ek + erange[1]*ep

    # assume an even number of pixels on the detector, seems reasonable
    ek_axis = np.linspace(ek_low, ek_high, nx_bins)

    # we need the arange as well as 2d array
    # arange was defined in the igor procedure Calculate_Da_values
    # it seems to be a constant, written in the calib2d file header
    # I decided to rename from "AzimuthLow"
    angle_low = arange[0]*1.2
    angle_high = arange[1]*1.2

    # check the effect of the additional range x1.2;
    # this is present in the igor code
    angle_axis = np.linspace(angle_low, angle_high, ny_bins)
    # the original program defines 2 waves,
    mcp_position_mm_matrix = np.zeros([nx_bins, ny_bins])
    angular_correction_matrix = np.zeros([nx_bins, ny_bins])
    e_correction = np.zeros(ek_axis.shape)
    # let's create a meshgrid for vectorized evaluation
    ek_mesh, angle_mesh = np.meshgrid(ek_axis, angle_axis)
    mcp_position_mm_matrix = mcp_position_mm(
        ek_mesh, angle_mesh,
        scanparameters,
    )
    Ang_Offset_px = 0  # add as optional input?
    E_Offset_px = 0  # add as optional input?
    angular_correction_matrix = (
        mcp_position_mm_matrix/magnification
        / (pixelsize*binning)
        + ny_bins/2
        + Ang_Offset_px
    )
    e_correction = (
        (
            ek_axis -
            ek*np.ones(ek_axis.shape)
        )
        / ep/de1/magnification/(pixelsize*binning)
        + nx_bins/2
        + E_Offset_px
    )
    w_dyde = np.gradient(angular_correction_matrix, ek_axis, axis=1)
    w_dyda = np.gradient(angular_correction_matrix, angle_axis, axis=0)
    w_dxda = 0
    w_dxde = np.gradient(e_correction, ek_axis, axis=0)
    jacobian_determinant = np.abs(w_dxde*w_dyda - w_dyde*w_dxda)

    return (
        ek_axis, angle_axis, angular_correction_matrix,
        e_correction, jacobian_determinant,
    )


def physical_unit_data(
    raw_data,
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
    e_correction_expand = np.ones(angular_correction_matrix.shape)*e_correction

    # Create a list of e and angle coordinates where to
    # evaluate the interpolating
    # function

    coords = (angular_correction_matrix.flatten(),
              e_correction_expand.flatten())
    # these coords seems to be pixels..

    x_bins = np.arange(0, raw_data.shape[0], 1)
    y_bins = np.arange(0, raw_data.shape[1], 1)

    # create interpolation function
    my_interpolating_function = RegularGridInterpolator(
        (x_bins, y_bins),
        raw_data,
        method='nearest',
        bounds_error=False,
        fill_value=33,
    )
    corrected_data = (
        np.reshape(
            my_interpolating_function(coords),
            angular_correction_matrix.shape,
        ) *
        jacobian_determinant
    )

    return corrected_data
