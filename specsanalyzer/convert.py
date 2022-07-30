import json

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
        ....

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
    h=open(file)
    Lines=h.readlines()
    start=0
    end=3
    step=1
    jArray=np.arange(start,end+step,step)
    endn=2

    nArray=np.arange(start,endn+step,step)

    endm=1

    mArray=np.arange(start,endm+step,step)

    endl=31

    lArray=np.arange(start,endl+step,step)

    ends=51

    sArray=np.arange(start,ends + step,step)

    endw=124

    wArray=np.arange(start,endw + step,step)

    endz = 32

    zArray=np.arange(start,endz + step,step)


    endp=25

    pArray=np.arange(start,endp+step,step)


    #Here we take the aInner, Das and RR value for the Low angle mode

    VLowAng = []

    for line_index in lArray:


        for j in jArray:
            A=7
            h=27 + A*line_index
            Value = [Lines[h + j]]

            value_as_dict = []
            for line in Value:
                name, equal, num1, num2, num3, diez, unit = line.split()
                val= {}
                val["name"] = name
                val["equal"] = equal
                val["num1"] = num1
                val["num2"] = num2
                val["num3"] = num3
                val["diez"] = diez
                val["unit"] = unit


            M=25 + A*line_index
            Val = Lines[M][22:27]
            param = json.loads(Val)


            N=26 + A*line_index
            Valu = [Lines[N]]
            Valu_as_dict=[]
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu={}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit


            V=json.loads(num),json.loads(num1),json.loads(num2),json.loads(num3),param
            VLowAng.append(V)


         #Here we take the aInner, Das and RR value for the med angle mode


    VMedAng=[]

    for line_index in wArray:


        for n in nArray:
            A=6
            h=259 + A*line_index
            Value = [Lines[h + n]]

            value_as_dict = []
            for line in Value:
                        name, equal, num1, num2, num3, diez, unit = line.split()
                        val= {}
                        val["name"] = name
                        val["equal"] = equal
                        val["num1"] = num1
                        val["num2"] = num2
                        val["num3"] = num3
                        val["diez"] = diez
                        val["unit"] = unit


            M=257 + A*line_index
            Val = Lines[M][25:30]
            param = json.loads(Val)


            N=258 + A*line_index
            Valu = [Lines[N]]
            Valu_as_dict=[]
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu={}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit


            V=json.loads(num),json.loads(num1),json.loads(num2),json.loads(num3),param
            VMedAng.append(V)



        #Here we take the aInner, Das and RR value for the high angle mode


    VHiAng=[]

    for line_index in sArray:


        for m in mArray:
            A=5
            h=1019 + A*line_index
            Value = [Lines[h + m]]

            value_as_dict = []
            for line in Value:
                        name, equal, num1, num2, num3, diez, unit = line.split()
                        val= {}
                        val["name"] = name
                        val["equal"] = equal
                        val["num1"] = num1
                        val["num2"] = num2
                        val["num3"] = num3
                        val["diez"] = diez
                        val["unit"] = unit


            M=1017 + A*line_index
            Val = Lines[M][23:28]
            param = json.loads(Val)



            N=1018 + A*line_index
            Valu = [Lines[N]]
            Valu_as_dict=[]
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu={}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit


            V=json.loads(num),json.loads(num1),json.loads(num2),json.loads(num3),param
            VHiAng.append(V)



    #Here we take the aInner, Das and RR value for the Wide angle mode


    VWideAng=[]

    for line_index in zArray:


        for j in jArray:
            A=7
            h=1287 + A*line_index
            Value = [Lines[h + j]]

            value_as_dict = []
            for line in Value:
                        name, equal, num1, num2, num3, diez, unit = line.split()
                        val= {}
                        val["name"] = name
                        val["equal"] = equal
                        val["num1"] = num1
                        val["num2"] = num2
                        val["num3"] = num3
                        val["diez"] = diez
                        val["unit"] = unit


            M=1285 + A*line_index
            Val = Lines[M][15:19]
            param = json.loads(Val)


            N=1286 + A*line_index
            Valu = [Lines[N]]
            Valu_as_dict=[]
            for lin in Valu:
                name, equal, num, diez, unit = lin.split()
                Valu={}
                Valu["name"] = name
                Valu["equal"] = equal
                Valu["num"] = num
                Valu["diez"] = diez
                Valu["unit"] = unit


            V=json.loads(num),json.loads(num1),json.loads(num2),json.loads(num3),param
            VWideAng.append(V)

    return np.concatenate((VLowAng,VMedAng,VHiAng,VWideAng),axis=0)
#Here we return all the value depending on the LensMode in a matrix
#preliminary functions written by Adrien, to be used until a suitable dictionary is created
#getparameters accepts as input the infofilename and calib2dfilename and returns an interpolated
# version of Da coeffiecients, not fully tested


def LensMode(calibfile,x):

    if x == "LensMode:LowAngularDispersion\n":
        h = parameters_table(calibfile)[0:128,0:5]
    if x == "LensMode:MediumAngularDispersion":
        h = parameters_table(calibfile)[128:502,0:5]
    if x == "LensMode:HighAngularDispersion\n":
        h = parameters_table(calibfile)[503:606,0:5]
    if x == "LensMode:WideAngleMode\n":
        h = parameters_table(calibfile)[607:738,0:5]
    return h


def closest_valueRR(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def GetParameters(infofile,calibfile):

    g = open(infofile)
    p= g.readlines()

    #We take the value of kinetic energy and pass energy from our file

    KE=json.loads(p[13][14:20])
    PE= json.loads(p[14][11:20])

    #We calculate the retard ratio RR

    RR = KE/PE

    start=0
    step=1
    endp=23
    pArray=(start,endp+step,step)

    x=p[3] #take the LensMode value


    #this function return the aInner,Das and RR value depending on our LensMode


    #print("Our aInner, Das, and RR parameters depending on the LensMode:",LensMode(x))
    #Define an array with lenght depending on which LensMode we choose to stock Das and RR value after

    stepInt = 4
    endInt = len(LensMode(calibfile,x))
    IntArray=np.arange(start,endInt,stepInt)

    RRList=[]
    for b in IntArray:

            RRList.append(LensMode(calibfile,x)[b][4])


    #define an array with all the value of RR in the LensMode selected
    RRArray=np.array(RRList)
    #print("All the parameters RR value",RRArray)
    #
    list1 = RRArray

    num=RR
    RRCloseVal=closest_valueRR(list1,num)

    stepDa=4

    DaArray=np.arange(start,endInt,stepDa)


    #The following Das array take the value for each Da depending on the LensMode selected and also the closest RR value

    Da1List=[]
    testlensmode=LensMode(calibfile,x)
    for l in DaArray:
        if testlensmode[l][4] == RRCloseVal:
            Da1 = testlensmode[l][1],testlensmode[l][2],testlensmode[l][3]
            Da1List.append(Da1)

    Da1=np.array(Da1List)


    Da3List=[]

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            Da3 = testlensmode[j+1][1],testlensmode[j+1][2],testlensmode[j+1][3]
            Da3List.append(Da3)

    Da3=np.array(Da3List)



    Da5List=[]

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            Da5 = testlensmode[j+2][1],testlensmode[j+2][2],testlensmode[j+2][3]
            Da5List.append(Da5)

    Da5=np.array(Da5List)




    Da7List=[]

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            Da7 = testlensmode[j+3][1],testlensmode[j+3][2],testlensmode[j+3][3]
            Da7List.append(Da7)

    Da7=np.array(Da7List)

    aInnerVal=[]

    for j in DaArray:
        if testlensmode[j][4] == RRCloseVal:
            V=testlensmode[j][0]
            aInnerVal.append(V)
    aInner=aInnerVal[0]
    #print("Our aInner value depending on the RR value is:",aInner)


    #Create a 3x3 matrix with our DAs value

    currentdamatrix=np.concatenate((Da1,Da3,Da5,Da7),axis=0)
    #print("The matrix of Das parameters linked with the aInner and RR value is:")
    return  currentdamatrix


#main function to integrate
#Calculate_Da_values()
#	Calculate_Polynomial_Coef_Da()  ->done
#	Calculate_MatrixCorrection()
#	PhysicalUnits_Data(RawData, PhysicalUnitsData)


#the function get the tabulated Da coefficients
# given for points at (-5% 0% +5%) of the pass energy
# this is range is defined in the array eShift
def calculate_polynomial_coef_da(ek,ep,eshift,currentdamatrix):

    #get the Das from the damatrix
    #da1=currentdamatrix[0][:]
    #da3=currentdamatrix[1][:]
    #da5=currentdamatrix[2][:]
    #da7=currentdamatrix[3][:]

    #calcualte the energy values for each da, given the eshift
    da_energy=eshift*ep+ek*np.ones(eshift.shape)

    #create the polinomial coeffiecient matrix, each is a third order polinomial

    dapolymatrix=np.zeros(currentdamatrix.shape)

    for i in range(0,currentdamatrix.shape[0]):
        #igor uses the fit poly 3, which should be a parabola
        dapolymatrix[i][:]=np.polyfit(da_energy, currentdamatrix[i][:], 2).transpose()
    return dapolymatrix
#the function now returns a matrix of the fit coeffiecients, given the physical energy scale
#each line of the matrix is a set of coefficients for each of the dai corrections

def zinner(ek,angle,dapolymatrix):
    #poly(D1, Ek )*(Ang) + 10^-2*poly(D3, Ek )*(Ang)^3 + 10^-4*poly(D5, Ek )*(Ang)^5 + 10^-6*poly(D7, Ek )*(Ang)^7
    result=0
    for i in range(0,dapolymatrix.shape[0]):
        #igor uses the fit poly 3, which should be a parabola
        result=result+ 10**(-(2*i))*angle**(1+2*i)*np.polyval(dapolymatrix[i][:],ek)

    return result

def zinner_diff(ek,angle,dapolymatrix):
    #poly(D1, Ek ) + 3*10^-2*poly(D3, Ek )*(Ang)^2 + 5*10^-4*poly(D5, Ek )*(Ang)^4 + 7*10^-6*poly(D7, Ek )*(Ang)^6
    result=0
    for i in range(0,dapolymatrix.shape[0]):
        #igor uses the fit poly 3, which should be a parabola
        result=result+ (2*i+1)*10**(-(2*i))*angle**(2*i)*np.polyval(dapolymatrix[i][:],ek)

    return result

def mcp_position_mm(ek,angle,ainner,dapolymatrix):
    mask=np.greater_equal(np.abs(angle),ainner)
    result=np.zeros(angle.shape)#ideally has to be evaluated on a mesh
    result = np.where(mask, zinner(ek,angle,dapolymatrix), np.sign(angle)*zinner(ek,angle,dapolymatrix)+(abs(angle)-ainner)*zinner_diff(ek,angle,dapolymatrix))

    return result
