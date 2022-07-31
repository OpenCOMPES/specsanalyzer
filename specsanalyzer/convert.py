import json
import re
import numpy as np
import xarray as xr


def convert_image(
    raw_image: np.ndarray,
    pass_energy: float,
    kinetic_energy: float,
    lens_mode: int,
    binning: int,
    calibration_dict: dict = {},
    detector_voltage: int = np.NaN,
) -> xr.DataArray:
    """Converts raw image into physical unit coordinates.

    Args:


    Returns:

        ...
    """


# TODO: populate

# This function will return the closest RR value with respect to our value
# preliminary functions written by Adrien, to be used until a suitable dictionary is created
# parameters_table accepts as input the calib2dfilename and returns a 4d matrix of
# conversion parameters for various

def parameters_table(file):
    # Here we define the vectors that will contain the value of the file
    
    h = open(file)
    lines = h.readlines()
    start = 0
    end = 3
    step = 1
    
    jArray = np.arange(start, end+step, step)
    endn = 2
    nArray = np.arange(start, endn+step, step)
    endm = 1
    mArray = np.arange(start, endm+step, step)
    endl = 31
    lArray = np.arange(start, endl+step, step)
    ends = 51
    sArray = np.arange(start, ends + step, step)
    endw = 124
    wArray = np.arange(start, endw + step, step)
    endz = 32
    zArray = np.arange(start, endz + step, step)
    
    # Here we take the aInner, Das and RR value for the Low angle mode

    VLowAng = []

    for line_index in lArray:

        for j in jArray:
            A = 7
            h = 27 + A*line_index
            Value = [lines[h + j]]

            for line in Value:
                name, equal, num1, num2, num3, diez, unit = line.split()
                val = {}
                val["name"] = name
                val["equal"] = equal
                val["num1"] = num1
                val["num2"] = num2
                val["num3"] = num3
                val["diez"] = diez
                val["unit"] = unit

            M = 25 + A*line_index
            Val = lines[M][22:27]
            param = json.loads(Val)

            N = 26 + A*line_index
            Valu = [lines[N]]
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu = {}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit

            V = json.loads(num), json.loads(num1), json.loads(
                num2), json.loads(num3), param
            VLowAng.append(V)

         # Here we take the aInner, Das and RR value for the med angle mode

    VMedAng = []

    for line_index in wArray:

        for n in nArray:
            A = 6
            h = 259 + A*line_index
            Value = [lines[h + n]]

            value_as_dict = []
            for line in Value:
                name, equal, num1, num2, num3, diez, unit = line.split()
                val = {}
                val["name"] = name
                val["equal"] = equal
                val["num1"] = num1
                val["num2"] = num2
                val["num3"] = num3
                val["diez"] = diez
                val["unit"] = unit

            M = 257 + A*line_index
            Val = lines[M][25:30]
            param = json.loads(Val)

            N = 258 + A*line_index
            Valu = [lines[N]]
            Valu_as_dict = []
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu = {}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit

            V = json.loads(num), json.loads(num1), json.loads(
                num2), json.loads(num3), param
            VMedAng.append(V)

        # Here we take the aInner, Das and RR value for the high angle mode

    VHiAng = []

    for line_index in sArray:

        for m in mArray:
            A = 5
            h = 1019 + A*line_index
            Value = [lines[h + m]]

            value_as_dict = []
            for line in Value:
                name, equal, num1, num2, num3, diez, unit = line.split()
                val = {}
                val["name"] = name
                val["equal"] = equal
                val["num1"] = num1
                val["num2"] = num2
                val["num3"] = num3
                val["diez"] = diez
                val["unit"] = unit

            M = 1017 + A*line_index
            Val = lines[M][23:28]
            param = json.loads(Val)

            N = 1018 + A*line_index
            Valu = [lines[N]]
            Valu_as_dict = []
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu = {}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit

            V = json.loads(num), json.loads(num1), json.loads(
                num2), json.loads(num3), param
            VHiAng.append(V)

    # Here we take the aInner, Das and RR value for the Wide angle mode

    VWideAng = []

    for line_index in zArray:

        for j in jArray:
            A = 7
            h = 1287 + A*line_index
            Value = [lines[h + j]]

            value_as_dict = []
            for line in Value:
                name, equal, num1, num2, num3, diez, unit = line.split()
                val = {}
                val["name"] = name
                val["equal"] = equal
                val["num1"] = num1
                val["num2"] = num2
                val["num3"] = num3
                val["diez"] = diez
                val["unit"] = unit

            M = 1285 + A*line_index
            Val = lines[M][15:19]
            param = json.loads(Val)

            N = 1286 + A*line_index
            Valu = [lines[N]]
            Valu_as_dict = []
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu = {}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit

            V = json.loads(num), json.loads(num1), json.loads(
                num2), json.loads(num3), param
            VWideAng.append(V)
    #These appear to be lists.. let's convert to array
    lad=np.array(VLowAng)  #LowAngularDispersion
    mad=np.array(VMedAng)  #MediumAngularDispersion
    had=np.array(VHiAng)   #HighAngularDispersion
    wam=np.array(VWideAng)  #WideAngleMode

    print(lad.shape,mad.shape,had.shape,wam.shape)

    print(np.concatenate((VLowAng, VMedAng, VHiAng, VWideAng), axis=0))
    
    # michele edit
    rr_array_lad=get_rr_da(lines,'LowAngularDispersion')
    rr_array_mad=get_rr_da(lines,'MediumAngularDispersion')
    rr_array_had=get_rr_da(lines,'HighAngularDispersion')
    rr_array_wam=get_rr_da(lines,'WideAngleMode')
    
    #this is the corre
    
    return np.concatenate((VLowAng, VMedAng, VHiAng, VWideAng), axis=0)
# Here we return all the value depending on the LensMode in a matrix
# preliminary functions written by Adrien, to be used until a suitable dictionary is created
# getparameters accepts as input the infofilename and calib2dfilename and returns an interpolated
# version of Da coeffiecients, not fully tested

def lens_mode(calibfile, x):

    if x == "LensMode:LowAngularDispersion\n":
        h = parameters_table(calibfile)[0:128, 0:5]
    if x == "LensMode:MediumAngularDispersion":
        h = parameters_table(calibfile)[128:502, 0:5]
    if x == "LensMode:HighAngularDispersion\n":
        h = parameters_table(calibfile)[503:606, 0:5]
    if x == "LensMode:WideAngleMode\n":
        h = parameters_table(calibfile)[607:738, 0:5]
    return h


def closest_value_rr(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def get_parameters(infofile, calibfile):

    g = open(infofile)
    p = g.readlines()

    # We take the value of kinetic energy and pass energy from our file
    KE = json.loads(p[13][14:20])
    PE = json.loads(p[14][11:20])

    # We calculate the retard ratio RR
    RR = KE/PE
    start = 0
    x = p[3]  # take the LensMode value
 
    # this function return the aInner,Das and RR value depending on our LensMode
    #print("Our aInner, Das, and RR parameters depending on the LensMode:",LensMode(x))
    # Define an array with lenght depending on which LensMode we choose to stock Das and RR value after

    stepInt = 4
    endInt = len(lens_mode(calibfile, x))
    IntArray = np.arange(start, endInt, stepInt)

    RRList = []
    for b in IntArray:

        RRList.append(lens_mode(calibfile, x)[b][4])

    # define an array with all the value of RR in the LensMode selected
    RRArray = np.array(RRList)
    #print("All the parameters RR value",RRArray)
    #
    list1 = RRArray

    num = RR
    RRCloseVal = closest_value_rr(list1, num)

    stepDa = 4

    DaArray = np.arange(start, endInt, stepDa)

    # The following Das array take the value for each Da depending on the LensMode selected and also the closest RR value

    Da1List = []
    testlensmode = lens_mode(calibfile, x)
    for l in DaArray:
        if testlensmode[l][4] == RRCloseVal:
            Da1 = testlensmode[l][1], testlensmode[l][2], testlensmode[l][3]
            Da1List.append(Da1)

    Da1 = np.array(Da1List)

    Da3List = []

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            Da3 = testlensmode[j+1][1], testlensmode[j+1][2], testlensmode[j+1][3]
            Da3List.append(Da3)

    Da3 = np.array(Da3List)

    Da5List = []

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            Da5 = testlensmode[j+2][1], testlensmode[j+2][2], testlensmode[j+2][3]
            Da5List.append(Da5)

    Da5 = np.array(Da5List)

    Da7List = []

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            Da7 = testlensmode[j+3][1], testlensmode[j+3][2], testlensmode[j+3][3]
            Da7List.append(Da7)

    Da7 = np.array(Da7List)

    aInnerVal = []

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            V = testlensmode[j][0]
            aInnerVal.append(V)
    aInner = aInnerVal[0]
    #print("Our aInner value depending on the RR value is:",aInner)

    # Create a 3x3 matrix with our DAs value

    currentdamatrix = np.concatenate((Da1, Da3, Da5, Da7), axis=0)
    #print("The matrix of Das parameters linked with the aInner and RR value is:")
    
    # michele edit, we also need a list of rr values correspondeing to the 
    
    
    
    
    return currentdamatrix


# main function to integrate
# Calculate_Da_values()
#	Calculate_Polynomial_Coef_Da()  ->done
#	Calculate_MatrixCorrection()
#	PhysicalUnits_Data(RawData, PhysicalUnitsData)


# the function get the tabulated Da coefficients
# given for points at (-5% 0% +5%) of the pass energy
# this is range is defined in the array eShift
def calculate_polynomial_coef_da(ek, ep, eshift, currentdamatrix):

    # get the Das from the damatrix
    # da1=currentdamatrix[0][:]
    # da3=currentdamatrix[1][:]
    # da5=currentdamatrix[2][:]
    # da7=currentdamatrix[3][:]

    # calcualte the energy values for each da, given the eshift
    da_energy = eshift*ep+ek*np.ones(eshift.shape)

    # create the polinomial coeffiecient matrix,
    # each is a third order polinomial

    dapolymatrix = np.zeros(currentdamatrix.shape)

    for i in range(0, currentdamatrix.shape[0]):
        # igor uses the fit poly 3, which should be a parabola
        dapolymatrix[i][:] = np.polyfit(da_energy,
                                        currentdamatrix[i][:], 2).transpose()
    return dapolymatrix
# the function now returns a matrix of the fit coeffiecients,
# given the physical energy scale
# each line of the matrix is a set of coefficients for each of the
# dai corrections


def zinner(ek,angle,dapolymatrix):
    #poly(D1, Ek )*(Ang) + 10^-2*poly(D3, Ek )*(Ang)^3 + 
    # 10^-4*poly(D5, Ek )*(Ang)^5 + 10^-6*poly(D7, Ek )*(Ang)^7
    out = 0
    for i in range(0,dapolymatrix.shape[0]):
        out = out + ((10**(-2*i))*
                     (angle**(1+2*i))*
                     np.polyval(dapolymatrix[i][:],ek))
    return out

def zinner_diff(ek,angle,dapolymatrix):
    # poly(D1, Ek ) + 3*10^-2*poly(D3, Ek )*(Ang)^2
    # + 5*10^-4*poly(D5, Ek )*(Ang)^4 + 7*10^-6*poly(D7, Ek )*(Ang)^6
    
    out = 0
    
    for i in range(0,dapolymatrix.shape[0]):
        
        out = out + ((10**(-2*i))*
                     (2*(i+1))*
                     (angle**(2*i))*
                     np.polyval(dapolymatrix[i][:],ek))
        
    return out


def mcp_position_mm(ek,angle,ainner,dapolymatrix):
    
    mask=np.less_equal(np.abs(angle),ainner)
   
    # result=np.zeros(angle.shape)#ideally has to be evaluated on a mesh

    ainner_vec=np.ones(angle.shape)*ainner
    
    result = np.where(mask,
                      zinner(ek,angle,dapolymatrix),
                      np.sign(angle)*zinner(ek,angle,dapolymatrix)+
                      (np.abs(angle)-ainner_vec)*
                      zinner_diff(ek,angle,dapolymatrix))
   
    return result


def get_scanparameters_fromcalib2d(infofilename,calib2dfilename):
    try:
        infofile=open(infofilename, "r")
        calibfile=open(calib2dfilename, "r")
        
        # now read the infofile and return a dictionary
        infodict = dict(get_pair(line) for line in infofile)
        # now from the retardatio ratio and lens mode go through the calib file 
        # and get the good stuff
        ek=float(infodict['KineticEnergy'])
        ep=float(infodict['PassEnergy'])
        lensmode=infodict['LensMode']
        rr=ek/ep
        
        #given the lens mode get all the retardatio ratios available
        calibfilelines=calibfile.readlines()
        rr_vec, damatrix_full = get_rr_da(calibfilelines,lensmode)
        closest_rr_index=bisection(rr_vec,rr)
        
        # return as the closest rr index the smallest in case of -1 output
        if closest_rr_index==-1:
            closest_rr_index=0
               
        # now compare the distance with the neighboring indexes, 
        # we need the second nearest rr 
        second_closest_rr_index=second_closest_rr(rr,rr_vec,closest_rr_index)
        
        
        
        damatrix=rr_vec
        return damatrix
    except IOError:
        print("Error: File does not appear to exist.")
        return 0

#Auxiliary function to load the info file
def get_pair(line):
    key, sep, value = line.strip().partition(":")
    return key, value

# Auxiliary function to find the closest rr index
# from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl


def second_closest_rr(rr,rrvec,closest_rr_index):
    if closest_rr_index==0 :
            second_closest_rr_index=1
    else:
        if closest_rr_index==(rrvec.size-1):
            second_closest_rr_index=closest_rr_index-1    
        else:
            # we are not at the edges, compare the neighbors and get the
            # closest    
            if (rr<rrvec[closest_rr_index]):
                second_closest_rr_index=closest_rr_index-1
            else:
                second_closest_rr_index=closest_rr_index+1    
                    
    return second_closest_rr_index

# this function should get both the rr array, and the corresponding Da matrices
# for a certain Angle mode
def get_da_block(lines,blockstart,blocklenght):
    damatrix=np.zeros([5,3])
    for p,q in enumerate(range(blockstart+1,blockstart+1+blocklenght,1)):
        
        linesplit=lines[q].split(" ")
        
        #print(linesplit) 
        
        if linesplit[0]=="aInner":
            damatrix[p][:]=float(linesplit[2])
        else:
            for i,j in enumerate(range(2,5,1)):
                damatrix[p][i]=float(linesplit[j])
    return damatrix

def get_rr_da(lines,modestring):
    rr=[]
    
    if modestring=="WideAngleMode":
        lines_idx= [i for i, item in enumerate(lines)
                    if ( re.search(modestring, item) and 
                        not(re.search("Super", item) ) )]
    else: 
        lines_idx= [i for i, item in enumerate(lines) 
                    if re.search(modestring, item)]
    
    block_start_list=[]    
    for i in lines_idx:
        wamline=lines[i]
        
        if (wamline.find("@")!=-1):
            wamlinestrip=wamline[:-2].replace("]", "")
            wamlinestrip=wamlinestrip.replace("[", "")
            rr.append(float(wamlinestrip.split("@")[1]))
            block_start_list.append(i)
        

    # here we should have a get block function
    
    #maybe dangerous?
    if len(block_start_list)>1:
        block_lenght=block_start_list[1]-block_start_list[0]-2
    else: 
        block_lenght=2
        
    rr_array=np.array(rr)
    # here we make a (rr lenght)x5x3 matrix 
    da_matrix_full=np.zeros([rr_array.shape[0],5,3])
    
    for i,start_index in enumerate(block_start_list):

        block_da_matrix=get_da_block(lines,start_index,block_lenght)
 
        da_matrix_full[i][:][:]=block_da_matrix[:][:]
         
    return rr_array, da_matrix_full
