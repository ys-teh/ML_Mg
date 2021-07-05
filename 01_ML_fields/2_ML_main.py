#/usr/bin/env python3

#################################################################################
## Train neural net
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
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
def load_data(YdataType,saveDir,dataDir,nPCs,trainSize):

    # Create directory for preprocessing results
    create_directory(saveDir)

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
def preprocess_data(saveDir,nPCs,trainX,trainY,nfft):

    # Keep a copy of the original data
    evaluatePCA = 0
    if evaluatePCA==1:
        originalTrainY = np.copy(trainY)

    # Perform PCA on Y data
    pca = PCA(n_components = nPCs)
    trainY = pca.fit_transform(trainY)

    # Evaluate PCA errors
    # Relative squared errors (RSE)
    # Normalized root-mean-square error (NRMSE)
    if evaluatePCA==1:
        # Evaluate errors on train set
        dataSize = np.shape(originalTrainY)[0]
        components = pca.components_
        mean = np.matmul(np.ones((dataSize,1)),np.asmatrix(pca.mean_))
        meanRSE = np.empty((nPCs))
        maxRSE = np.empty(nPCs)
        meanNRMSE = np.empty((nPCs))
        maxNRMSE = np.empty((nPCs))
        for iPC in range(nPCs):
            residue = originalTrainY - mean - np.matmul(trainY[:,0:iPC+1],components[0:iPC+1,:])
            RSE = np.empty(dataSize)
            NRMSE = np.empty(dataSize)
            for iData in range(dataSize):
                RSE[iData] = LA.norm(residue[iData])/LA.norm(originalTrainY[iData])
                den = np.max(originalTrainY[iData])-np.min(originalTrainY[iData])
                NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
            meanRSE[iPC] = np.mean(RSE)
            maxRSE[iPC] = np.amax(RSE)
            meanNRMSE[iPC] = np.mean(NRMSE)
            maxNRMSE[iPC] = np.amax(NRMSE)
        print("PCA mean RSE (train set): ",meanRSE)
        print("PCA max RSE (train set): ",maxRSE)
        print("PCA mean NRMSE (train set): ",meanNRMSE)
        print("PCA max NRMSE (train set): ",maxNRMSE)

    # Compute sum of variance ratio 
    varianceRatio = pca.explained_variance_ratio_
    sumVarianceRatio = np.empty((nPCs))
    sumVarianceRatio[0] = varianceRatio[0]
    for iPC in range(1,nPCs):
        sumVarianceRatio[iPC] = varianceRatio[iPC] + sumVarianceRatio[iPC-1]

    # Save PCA results
    joblib.dump(pca,saveDir+'pca')
    np.savetxt(saveDir+'pca_explained_variance.txt',pca.explained_variance_,delimiter=',',fmt='%1.5e')
    np.savetxt(saveDir+'pca_explained_variance_ratio.txt',varianceRatio,delimiter=',',fmt='%1.5e')
    np.savetxt(saveDir+'pca_explained_variance_ratio_sum.txt',sumVarianceRatio,delimiter=',',fmt='%1.5e')

    # Rescale data
    scalerX = StandardScaler()
    trainX = scalerX.fit_transform(trainX)
    scalerY = StandardScaler()
    trainY = scalerY.fit_transform(trainY)

    # Save scaler
    joblib.dump(scalerX,saveDir+'scalerX')
    joblib.dump(scalerY,saveDir+'scalerY')

    return trainX,trainY

##-------------------------------------------------------------------------------
## Machine learning model
## nInDim = input dimension of model
## nPCs = number of principal components = output dimension of model
##-------------------------------------------------------------------------------
def build_model(nInDim,nPCs):
    nHidden1 = 500
    nHidden2 = 500
    model = keras.Sequential([
        Dense(nHidden1, activation='tanh', input_shape=[nInDim], kernel_regularizer=l2(0.0001)),
        Dense(nHidden2, activation='tanh', kernel_regularizer=l2(0.0001)),
        Dense(nPCs)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

    return model

##-------------------------------------------------------------------------------
## Evaluate ML test results
##-------------------------------------------------------------------------------
def evaluate_test_results(model,YdataType,saveDir,dataDir,testX,testY,nfft):

    # Load scalers
    scalerX = joblib.load(saveDir+'scalerX')
    scalerY = joblib.load(saveDir+'scalerY')

    # Evaluate and rescale
    originalTestX = np.copy(testX)
    testX = scalerX.transform(testX)
    predY = model.predict(testX)
    predY = scalerY.inverse_transform(predY)

    # Reconstruct using principal components
    pca = joblib.load(saveDir+'pca')
    predY = pca.inverse_transform(predY)

    # Save some prediction results
    with open(saveDir+'prediction_sample.pickle' ,'wb') as f:
        pickle.dump([originalTestX[2],testY[2],predY[2]], f)

    # Compute relative error (RE) and maximum absolute error (MAE)
    residue = predY - testY
    dataSize = np.shape(predY)[0]
    RSE = np.empty((dataSize))
    NRMSE = np.empty((dataSize))
    MAE = np.empty((dataSize))
    RMSE = np.empty((dataSize)) # Needed for 4_compute_DEN_RMSE_for_comparison.py
    for iData in range(dataSize):
        RSE[iData] = LA.norm(residue[iData])/LA.norm(testY[iData])
        den = np.max(testY[iData])-np.min(testY[iData])
        NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
        MAE[iData] = np.amax(np.abs(residue[iData]))
        RMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)
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
    np.savetxt(saveDir+'ml_test_rse.txt',RSE)
    np.savetxt(saveDir+'ml_test_nrmse.txt',NRMSE)
    np.savetxt(saveDir+'ml_test_mae.txt',MAE)
    np.savetxt(saveDir+'ml_test_rmse.txt',RMSE)

    # Evaluate PCA errors
    evaluatePCA = 1
    if evaluatePCA==1:
        originalTestY = np.copy(testY)

        # Evaluate errors on test set
        testY = pca.transform(testY)
        testY = pca.inverse_transform(testY)
        residue = testY - originalTestY
        RSE = np.empty((dataSize))
        NRMSE = np.empty((dataSize))
        MAE = np.empty((dataSize))
        for iData in range(dataSize):
            RSE[iData] = LA.norm(residue[iData])/LA.norm(originalTestY[iData])
            den = np.max(testY[iData])-np.min(testY[iData])
            NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
            MAE[iData] = np.amax(np.abs(residue[iData]))
        np.savetxt(saveDir+'pca_test_rse.txt',RSE)
        np.savetxt(saveDir+'pca_test_nrmse.txt',NRMSE)
        np.savetxt(saveDir+'pca_test_mae.txt',MAE)

    # Save predY
    np.save(saveDir+'predY.npy',predY)

    return

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Set parameters
    YdataType = 'DEN'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
    trainSize = int(sys.argv[1])
    print('Train size is ',trainSize)
    nPCs = 50                # Number of principal components  
    nInDim = 6
    nfft = 36*64*60
    dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'
    saveDir = 'Saved_ml_results_occopt4_train'+str(trainSize)+'/'+YdataType+'/'

    # Time
    time0 = time.time()

    # Load raw data
    print("Loading data...")
    trainX,testX,trainY,testY = load_data(YdataType,saveDir,dataDir,nPCs,trainSize)

    # Preprocessing and perform ML
    trainNN = 0
    if trainNN==1:
        print("Preprocessing data...")
        trainX,trainY = preprocess_data(saveDir,nPCs,trainX,trainY,nfft)
        print("Running ML...")
        model = build_model(nInDim,nPCs)
        model.summary()
        history = model.fit(trainX,trainY,batch_size=128,epochs=4000,verbose=0)
        model.save(saveDir+'ml_model.h5')
        #model = LR().fit(trainX,trainY)
    else:
        print("Loading ML model...")
        model = keras.models.load_model(saveDir+'ml_model.h5')

    # Evaluate test errors (Note testY has not undergone preprocessing)
    evaluate_test_results(model,YdataType,saveDir,dataDir,testX,testY,nfft)

    # Save testX
    np.save(saveDir.split('/')[0]+'/testX.npy', testX)

    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

if __name__ == '__main__': main()
