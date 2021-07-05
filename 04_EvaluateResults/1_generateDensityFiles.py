#!/usr/bin/env python3

#################################################################################
## Dataset: Mg
## Generate electron density files
## Create by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
import os
import math

##-------------------------------------------------------------------------------
## Load and scale data
##-------------------------------------------------------------------------------
def unnormalize_electron_density(density, abcrst):
    # Check that predicted sum of electrons is 8.0
    if abs(np.sum(density)-8.0)>1e-6:
        print("Error: Sum of electrons is %f!" %(np.sum(density)))

    # Variables from Abinit
    nfft = 36*64*60
    #acell = [6.0419031073,10.464702002,9.8453372801]
    acell = [5.89,10.201779256580686,9.56]

    # Strain values
    a = abcrst[0]
    b = abcrst[1]
    c = abcrst[2]
    r = abcrst[3]
    s = abcrst[4]
    t = abcrst[5]
    cos_r = math.cos(math.radians(r))
    cos_s = math.cos(math.radians(s))
    cos_t = math.cos(math.radians(t))
    dV = (acell[0]*acell[1]*acell[2]*a*b*c)/nfft \
         * math.sqrt( 1 + 2*cos_r*cos_s*cos_t - cos_r*cos_r - cos_s*cos_s - cos_t*cos_t )
    density /= dV

    return density

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():
    testX = np.load('../01_ML_fields/Saved_ml_results_occopt4_train2000/testX.npy')
    predY = np.load('../01_ML_fields/Saved_ml_results_occopt4_train2000/DEN/predY.npy')
    testSize = np.shape(testX)[0]
    
    dirName = 'Electron_density_files/'
    try:
        os.makedirs(dirName)
    except:
        if not os.path.isdir(dirName):
            raise

    for index in range(testSize):
        print(index)
        abcrst = testX[index,:]
        density = predY[index,:]
        density = unnormalize_electron_density(density, abcrst)
        np.savetxt(dirName+'pred'+str(index+1)+'.dat',density) 

if __name__ == '__main__': main()
