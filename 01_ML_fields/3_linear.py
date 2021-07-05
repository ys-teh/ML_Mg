#/usr/bin/env python3

#################################################################################
## Construct PCA/linear regression model instead of PCA/neural net.
## Created by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
from numpy import linalg as LA
import math
import os,sys
import joblib
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression as LR
import pickle

#--------------------------------------------------------------------------------
# Create directory
#--------------------------------------------------------------------------------
def create_directory(path):
    try:
        os.makedirs(path)
    except:
        if not os.path.isdir(path):
            raise

    return

#--------------------------------------------------------------------------------
# Load raw data
#--------------------------------------------------------------------------------
def load_data(YdataType,dataDir,nPCs,trainSize):

    # Load data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)

    trainX = X[:trainSize]
    testX = X[2000:3000]
    trainY = Y[:trainSize]
    testY = Y[2000:3000]

    return trainX,testX,trainY,testY

#--------------------------------------------------------------------------------
# Load raw data, perform PCA and rescale data
#--------------------------------------------------------------------------------
def preprocess_data(nPCs,trainX,trainY,nfft):

    # Perform PCA on Y data
    pca = PCA(n_components = nPCs)
    trainY = pca.fit_transform(trainY)

    # Rescale data
    scalerX = StandardScaler()
    trainX = scalerX.fit_transform(trainX)
    scalerY = StandardScaler()
    trainY = scalerY.fit_transform(trainY)

    return trainX,trainY,pca,scalerX,scalerY

##-------------------------------------------------------------------------------
## Evaluate ML test results
##-------------------------------------------------------------------------------
def evaluate_test_results(model,YdataType,saveDir,testX,testY,pca,scalerX,scalerY,nfft):

    # Evaluate and rescale
    originalTestX = np.copy(testX)
    testX = scalerX.transform(testX)
    predY = model.predict(testX)
    predY = scalerY.inverse_transform(predY)

    # Reconstruct using principal components
    predY = pca.inverse_transform(predY)

    # Compute relative error (RE) and maximum absolute error (MAE)
    residue = predY - testY
    dataSize = np.shape(predY)[0]
    RSE = np.empty((dataSize))
    NRMSE = np.empty((dataSize))
    MAE = np.empty((dataSize))
    for iData in range(dataSize):
        RSE[iData] = LA.norm(residue[iData])/LA.norm(testY[iData])
        den = np.max(testY[iData])-np.min(testY[iData])
        NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
        MAE[iData] = np.amax(np.abs(residue[iData]))
    meanRSE = np.mean(RSE)
    maxRSE = np.amax(RSE)
    meanNRMSE = np.mean(NRMSE)
    maxNRMSE = np.amax(NRMSE)
    meanMAE = np.mean(MAE)
    maxMAE = np.amax(MAE)
    print("Mean RSE: ",meanRSE)
    print("Max RSE: ",maxRSE)
    print("Mean NRMSE: ",meanNRMSE)
    print("Max NRMSE: ",maxNRMSE)
    print("Mean MAE: ",meanMAE)
    print("Max MAE: ",maxMAE)
    np.savetxt(saveDir+'linear_test_rse.txt',RSE)
    np.savetxt(saveDir+'linear_test_nrmse.txt',NRMSE)
    np.savetxt(saveDir+'linear_test_mae.txt',MAE)

    return

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Set parameters
    YdataType = 'VCLMB'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
    trainSize = int(sys.argv[1])
    nPCs = 50                # Number of principal components  
    nInDim = 6
    nfft = 36*64*60
    dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'
    saveDir = 'Saved_ml_results_occopt4_train'+str(trainSize)+'/'+YdataType+'/'

    # Time
    time0 = time.time()

    # Load raw data
    print("Loading data...")
    trainX,testX,trainY,testY = load_data(YdataType,dataDir,nPCs,trainSize)

    # Preprocessing and perform ML
    print("Preprocessing data...")
    trainX,trainY,pca,scalerX,scalerY = preprocess_data(nPCs,trainX,trainY,nfft)
    print("Running linear regression...")
    model = LR().fit(trainX,trainY)

    # Evaluate test errors (Note testY has not undergone preprocessing)
    evaluate_test_results(model,YdataType,saveDir,testX,testY,pca,scalerX,scalerY,nfft)

    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

if __name__ == '__main__': main()
