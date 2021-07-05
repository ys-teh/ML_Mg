#/usr/bin/env python3

#################################################################################
# Create by Ying Shi Teh
# Last modified on July 5, 2021
#################################################################################

import numpy as np
from numpy import linalg as LA
import math
import os
import joblib
import time
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})
from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression as LR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import pickle

#--------------------------------------------------------------------------------
# Load strain
#--------------------------------------------------------------------------------
def load_strain():

    strain = np.load('Uniaxial_test_results/strain_direction2.npy')

    # Pick only 5 strain values
    strain = strain[[0,15,30,45,60],:]

    return strain

#--------------------------------------------------------------------------------
# Extract trueXRED
#--------------------------------------------------------------------------------
def extract_trueXRED():

    trueXREDy = np.zeros((21))
    for i in range(21):
        filename = 'Abinit_runs/direction2/mg' + str(i+1) + '.out'
        with open(filename) as inputFile:
            flag = False
            for line in inputFile:
                if 'END DATASET(S)' in line:
                    flag = True
                    continue
                elif flag and 'xred' in line: 
                    trueXREDy[i] = line.split()[2]
                    break

    return trueXREDy

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Variables from Abinit
    ngfft = [36,64,60]
    nfft = ngfft[0]*ngfft[1]*ngfft[2]
    acell = [5.89,10.201779256580686,9.56]

    # Directory where machine learning results are saved
    saveDirXRED = '../02_ML_XRED/Saved_ml_results_occopt4_train2000/XRED/'
    saveDirDEN = '../01_ML_fields/Saved_ml_results_occopt4_train2000/DEN/'

    # Extract true XRED values
    trueXREDy = extract_trueXRED()
    plt.figure()
    plt.plot(trueXREDy,'*-')
    plt.plot(1./6.-trueXREDy,'o-')
    plt.show()


if __name__ == '__main__': main()
