#/usr/bin/env python3

#################################################################################
## k-fold cross validation for PCA/neural net model
## Created by Ying Shi Teh
## Last modified on July 5, 2021
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
def load_data(YdataType,dataDir,nPCs):

    # Load data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)

    X = X[:2000]
    Y = Y[:2000]

    return X,Y

#--------------------------------------------------------------------------------
# Load raw data, perform PCA and rescale data
#--------------------------------------------------------------------------------
def preprocess_data(nPCs,trainX,testX,trainY,testY,nfft):

    # Keep a copy of the original data
    evaluatePCA = 0
    if evaluatePCA==1:
        originalTrainY = np.copy(trainY)
        originalTestY = np.copy(testY)

    # Perform PCA on Y data
    pca = PCA(n_components = nPCs)
    trainY = pca.fit_transform(trainY)
    testY = pca.transform(testY)

    # Evaluate PCA errors (normalized square errors)
    if evaluatePCA==1:
        # Evaluate errors on train set
        dataSize = np.shape(originalTrainY)[0]
        components = pca.components_
        mean = np.matmul(np.ones((dataSize,1)),np.asmatrix(pca.mean_))
        meanRSE = np.empty((nPCs))
        maxRSE = np.empty((nPCs))
        meanNRMSE = np.empty((nPCs))
        maxNRMSE = np.empty((nPCs))
        for iPC in range(nPCs):
            residue = originalTrainY - mean - np.matmul(trainY[:,0:iPC+1],components[0:iPC+1,:])
            RSE = np.empty((dataSize))
            NRMSE = np.empty((dataSize))
            for iData in range(0,dataSize):
                RSE[iData] = LA.norm(residue[iData])/LA.norm(originalTrainY[iData])
                den = np.max(originalTrainY[iData]) - np.min(originalTrainY[iData])
                NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
            meanRSE[iPC] = np.mean(RSE)
            maxRSE[iPC] = np.max(RSE)
            meanNRMSE[iPC] = np.mean(NRMSE)
            maxNRMSE[iPC] = np.max(NRMSE)
        print("PCA mean RSE (train set): ",meanRSE)
        print("PCA max RSE (train set): ",maxRSE)
        print("PCA mean NRMSE (train set): ",meanNRMSE)
        print("PCA max NRMSE (train set): ",maxNRMSE)
        # Evaluate errors on test set
        dataSize = np.shape(originalTestY)[0]
        mean = np.matmul(np.ones((dataSize,1)),np.asmatrix(pca.mean_))
        for iPC in range(nPCs):
            residue = originalTestY - mean - np.matmul(testY[:,0:iPC+1],components[0:iPC+1,:])
            RSE = np.empty(dataSize)
            for iData in range(0,dataSize):
                RSE[iData] = LA.norm(residue[iData])/LA.norm(originalTestY[iData])
                den = np.max(originalTestY[iData]) - np.min(originalTestY[iData])
                NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
            meanRSE[iPC] = np.mean(RSE)
            maxRSE[iPC] = np.max(RSE)
            meanNRMSE[iPC] = np.mean(NRMSE)
            maxNRMSE[iPC] = np.max(NRMSE)
        print("PCA mean RSE (test set): ",meanRSE)
        print("PCA max RSE (test set): ",maxRSE)
        print("PCA mean NRMSE (test set): ",meanNRMSE)
        print("PCA max NRMSE (test set): ",maxNRMSE)

    # Compute sum of variance ratio 
    varianceRatio = pca.explained_variance_ratio_
    sumVarianceRatio = np.empty((nPCs))
    sumVarianceRatio[0] = varianceRatio[0]
    for iPC in range(1,nPCs):
        sumVarianceRatio[iPC] = varianceRatio[iPC] + sumVarianceRatio[iPC-1]

    # Rescale data
    scalerX = StandardScaler()
    trainX = scalerX.fit_transform(trainX)
    testX = scalerX.transform(testX)
    scalerY = StandardScaler()
    trainY = scalerY.fit_transform(trainY)
    testY = scalerY.transform(testY)

    return trainX,testX,trainY,testY,pca,scalerX,scalerY

##-------------------------------------------------------------------------------
## Machine learning model
## nInDim = input dimension of model
## nPCs = number of principal components = output dimension of model
##-------------------------------------------------------------------------------
def build_model(nInDim,nPCs):
    nHidden1 = 2000
    nHidden2 = 2000
#    nHidden3 = 1000
    model = keras.Sequential([
        Dense(nHidden1, activation='tanh', input_shape=[nInDim], kernel_regularizer=l2(0.0001)),
        Dense(nHidden2, activation='tanh', kernel_regularizer=l2(0.0001)),
#        Dense(nHidden3, activation='tanh', kernel_regularizer=l2(0.0001)),
        Dense(nPCs)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

    return model

##-------------------------------------------------------------------------------
## Evaluate ML test results
##-------------------------------------------------------------------------------
def evaluate_test_results(model,YdataType,dataDir,nfft,pca,scalerX,scalerY,testID):

    # Load test data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)
    testX = X[testID]
    testY = Y[testID]

    # Evaluate and rescale
    testX = scalerX.transform(testX)
    predY = model.predict(testX)
    predY = scalerY.inverse_transform(predY)

    # Reconstruct using principal components
    predY = pca.inverse_transform(predY)

    # Compute normalized squared error
    residue = predY - testY
    dataSize = np.shape(predY)[0]
    RSE = np.empty((dataSize))
    NRMSE = np.empty((dataSize))
    MAE = np.empty((dataSize))
    for iData in range(dataSize):
        RSE[iData] = LA.norm(residue[iData])/LA.norm(testY[iData])
        den = np.max(testY[iData]) - np.min(testY[iData])
        NRMSE[iData] = LA.norm(residue[iData])/math.sqrt(nfft)/den
        MAE[iData] = np.max(np.abs(residue[iData]))
    meanRSE = np.mean(RSE)
    maxRSE = np.max(RSE)
    meanNRMSE = np.mean(NRMSE)
    maxNRMSE = np.max(NRMSE)
    meanMAE = np.mean(MAE)
    maxMAE = np.max(MAE)
    print("Mean RSE: ",meanRSE)
    print("Max RSE: ",maxRSE)
    print("Mean NRMSE: ",meanNRMSE)
    print("Max NRMSE: ",maxNRMSE)
    print("Mean MAE: ",meanMAE)
    print("Max MAE: ",maxMAE)

    return meanRSE,maxRSE,meanNRMSE,maxNRMSE,meanMAE,maxMAE

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Set parameters
    YdataType = 'DEN'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
    nInDim = 6
    nPCs = 30                # Number of principal components  
    nfft = 36*64*60
    dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'

    # Time
    time0 = time.time()

    # Load raw data
    print("Loading data...")
    X,Y = load_data(YdataType,dataDir,nPCs)

    # Perform k-fold cross-validation
    k  = 4
    allErrors = np.empty((k,6))
    kf = KFold(n_splits=k)
    fold = 0
    for trainID, testID in kf.split(X):
        #if fold == 1:
        #    break
        trainX, testX = X[trainID], X[testID]
        trainY, testY = Y[trainID], Y[testID]
        trainX,testX,trainY,testY,pca,scalerX,scalerY = preprocess_data(nPCs,trainX,testX,trainY,testY,nfft)

        # Perform ML
        print("Running ML...")
        model = build_model(nInDim,nPCs)
        model.summary()
        #callback = keras.callbacks.EarlyStopping(monitor='loss',patience=20,verbose=1,min_delta=0.0)
        history = model.fit(trainX,trainY,batch_size=128,epochs=4000,verbose=0,validation_data=(testX,testY)) #,callbacks=[callback])
    
        # Evaluate test errors
        meanRSE,maxRSE,meanNRMSE,maxNRMSE,meanMAE,maxMAE = evaluate_test_results(model,YdataType,dataDir,nfft,pca,scalerX,scalerY,testID)
        allErrors[fold] = [meanRSE,maxRSE,meanNRMSE,maxNRMSE,meanMAE,maxMAE]
        print("Loss: ",history.history['loss'][-1])
        print("Val loss: ",history.history['val_loss'][-1])

        fold += 1

    # Print average errors
    print("\n")
    print("Averaging results from all folds...")
    print("Mean RSE: ",np.mean(allErrors[:,0]))
    print("Max RSE: ",np.mean(allErrors[:,1]))
    print("Mean NRMSE: ",np.mean(allErrors[:,2]))
    print("Max NRMSE: ",np.mean(allErrors[:,3]))
    print("Mean MAE: ",np.mean(allErrors[:,4]))
    print("Max MAE: ",np.mean(allErrors[:,5]))

    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

    # Plot ML results
    print("Plotting results...")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

if __name__ == '__main__': main()
