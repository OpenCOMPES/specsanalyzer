import numpy as np
import scipy.interpolate as interp
import pandas as pd
import matplotlib.pyplot as plt
#import xarray as xr
#import cartopy.crs as ccrs
#import operator
import json
#from netCDF4 import Dataset
#from scipy.io import netcdf



#We have to insert as an input of the function ParametersTable the text file containing all the parameters, in our case this is phoibosEPFL.txt
#Then, the function ParametersTale will return all the Das value with the corresponding LensMode, aInner and RR parameters as a Matrix
file=input('Enter txt file containing tables parameters in the format "file.txt": ')

def ParametersTable(file):

    #Here we define the vectors that will contain the value of the file
    
    h=open(file,'r')
    
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
     
    VLowAng=[]
    
    for l in lArray:
        
        
        for j in jArray:
            A=7
            h=27 + A*l
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
               
            
            M=25 + A*l
            Val = Lines[M][22:27]
            param = json.loads(Val)


            N=26 + A*l
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
    
    for l in wArray:
        
        
        for n in nArray:
            A=6
            h=259 + A*l
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
                
            
            M=257 + A*l
            Val = Lines[M][25:30]
            param = json.loads(Val)
        
                
            N=258 + A*l
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
    
    for l in sArray:
        
        
        for m in mArray:
            A=5
            h=1019 + A*l
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
                        
            
            M=1017 + A*l
            Val = Lines[M][23:28]
            param = json.loads(Val)
      
                  
                
            N=1018 + A*l
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
    
    for l in zArray:
        
        
        for j in jArray:
            A=7
            h=1287 + A*l
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
                   
            
            M=1285 + A*l
            Val = Lines[M][15:19]
            param = json.loads(Val)
         
                
            N=1286 + A*l
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
            
            pd.options.display.max_rows=99999 
    return np.concatenate((VLowAng,VMedAng,VHiAng,VWideAng),axis=0)  #Here we return all the value depending on the LensMode in a matrix

print(ParametersTable(file))


##############################################################################


#Here, we have to insert as an input the txt file containing the specification of our instrument, in our case this is info.txt
#The function GetParameters will select the aInner, Das and RR value depending on the LensMode
#She also give use the RRArray that contain all the RR value that will be useful to compute the rr_factor with the interpolation
#The function also calculate the RR value corresponding on our measurement and give the cloest RR value this one
#Finally the function return the aInner value corresponding of the RR value and also a matrix with the corresponding Das

info=input('Enter txt file containing the detectors specification in the format "file.txt": ')


def GetParameters(info):
    
    g = open(info,'r')
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
    
    def LensMode(x):
        
        if x == "LensMode:LowAngularDispersion\n":
           h = ParametersTable(file)[0:128,0:5]
        if x == "LensMode:MediumAngularDispersion":
           h = ParametersTable(file)[128:502,0:5]
        if x == "LensMode:HighAngularDispersion\n":
           h = ParametersTable(file)[503:606,0:5]
        if x == "LensMode:WideAngleMode\n":
           h = ParametersTable(file)[607:738,0:5]
           return h
       
    print("Our aInner, Das, and RR parameters depending on the LensMode:",LensMode(x)) 
       
    
    
    #Define an array with lenght depending on which LensMode we choose to stock Das and RR value after
    stepInt = 4
    endInt = len(LensMode(x))
    IntArray=np.arange(start,endInt,stepInt)
    
    RRList=[]    
    for b in IntArray:
        
            RRList.append(LensMode(x)[b][4])
            
            
     #define an array with all the value of RR in the LensMode selected       
    RRArray=np.array(RRList)
    print("All the parameters RR value",RRArray)
    
    #This function will return the closest RR value with respect to our value
    
    def closest_valueRR(input_list, input_value):
     
      arr = np.asarray(input_list)
     
      i = (np.abs(arr - input_value)).argmin()  
     
      return arr[i]
     
    if __name__ == "__main__" :
     
       list1 = RRArray

       num=RR
     
       val=closest_valueRR(list1,num)
     
       print("The closest RR value to our value "+str(num)+" is",val)
       
       
    RRCloseVal=val

    stepDa=4

    DaArray=np.arange(start,endInt,stepDa)
    
    
    #The following Das array take the value for each Da depending on the LensMode selected and also the closest RR value
    
    Da1List=[]
    
    for l in DaArray:
        if LensMode(x)[l][4] == RRCloseVal:
            Da1 = LensMode(x)[l][1],LensMode(x)[l][2],LensMode(x)[l][3]
            Da1List.append(Da1)
            
    Da1=np.array(Da1List)
    
    
    Da3List=[]
    
    for j in DaArray:
        if LensMode(x)[j][4] == RRCloseVal:
            Da3 = LensMode(x)[j+1][1],LensMode(x)[j+1][2],LensMode(x)[j+1][3]
            Da3List.append(Da3)
            
    Da3=np.array(Da3List)
    
    
    
    Da5List=[]
    
    for j in DaArray:
        if LensMode(x)[j][4] == RRCloseVal:
            Da5 = LensMode(x)[j+2][1],LensMode(x)[j+2][2],LensMode(x)[j+2][3]
            Da5List.append(Da5)
            
    Da5=np.array(Da5List)
    
    
    
    
    Da7List=[]
    
    for j in DaArray:
        if LensMode(x)[j][4] == RRCloseVal:
            Da7 = LensMode(x)[j+3][1],LensMode(x)[j+3][2],LensMode(x)[j+3][3]
            Da7List.append(Da7)
            
    Da7=np.array(Da7List)
    
    aInnerVal=[]

    for j in DaArray:
        if LensMode(x)[j][4] == RRCloseVal:
            V=LensMode(x)[j][0]
            aInnerVal.append(V)
    aInner=aInnerVal[0]
    print("Our aInner value depending on the RR value is:",aInner)
    

    #Create a 3x3 matrix with our DAs value

    Matrix=np.concatenate((Da1,Da3,Da5,Da7),axis=0)
    print("The matrix of Das parameters linked with the aInner and RR value is:")
    return  Matrix

print(GetParameters(info))



