#/usr/bin/env python3

#################################################################################
# Construct linear regression model for ETOT (i.e. total free energy)
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
# Load raw data
#--------------------------------------------------------------------------------
def load_data(YdataType,dataDir,trainSize):

    # Load data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)

    trainX = X[:trainSize]
    trainY = Y[:trainSize]
    testX = X[2000:3000]
    testY = Y[2000:3000]

    return trainX,testX,trainY,testY

#--------------------------------------------------------------------------------
# Load raw data, perform PCA and rescale data
#--------------------------------------------------------------------------------
def preprocess_data(trainX,trainY):

    # Rescale data
    scalerX = StandardScaler()
    trainX = scalerX.fit_transform(trainX)
    scalerY = StandardScaler()
    trainY = scalerY.fit_transform(trainY)

    return trainX,trainY,scalerX,scalerY

##-------------------------------------------------------------------------------
## Evaluate ML test results
##-------------------------------------------------------------------------------
def evaluate_test_results(model,YdataType,saveDir,testX,testY,scalerX,scalerY):

    # Evaluate and rescale
    originalTestX = np.copy(testX)
    testX = scalerX.transform(testX)
    predY = model.predict(testX)
    predY = scalerY.inverse_transform(predY)

    # Compute relative error (RE), maximum absolute error (MAE),
    # displacement error (in Bohr) and true displacements (in Bohr)
    residue = np.absolute(predY - testY)
    relError = residue/np.absolute(testY)
    meanAbsError = np.mean(residue)
    maxAbsError = np.max(residue)
    meanRelError = np.mean(relError)
    maxRelError = np.max(relError)
    print("Mean absolute error (Hartree): ",meanAbsError)
    print("Max absolute error (Hartree): ",maxAbsError)
    print("Mean relative error: ",meanRelError)
    print("Max relative error: ",maxRelError)
    np.savetxt(saveDir+'linear_test_abs_error.txt',residue)
    np.savetxt(saveDir+'linear_test_relative_error.txt',relError)

    return

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Set parameters
    YdataType = 'ETOT'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
    trainSize = 2000
    nInDim = 6
    #acell = [6.0419031073,10.464702002,9.8453372801]
    acell = [5.89,10.201779256580686,9.56]
    dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'
    saveDir = 'Saved_ml_results_occopt4_train'+str(trainSize)+'/'+YdataType+'/'

    # Time
    time0 = time.time()

    # Load raw data
    print("Loading data...")
    trainX,testX,trainY,testY = load_data(YdataType,dataDir,trainSize)

    # Preprocessing and perform ML
    print("Preprocessing data...")
    trainX,trainY,scalerX,scalerY = preprocess_data(trainX,trainY)
    print("Running linear regression...")
    model = LR().fit(trainX,trainY)

    # Evaluate test errors (Note testY has not undergone preprocessing)
    evaluate_test_results(model,YdataType,saveDir,testX,testY,scalerX,scalerY)

    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

if __name__ == '__main__': main()
