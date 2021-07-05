#/usr/bin/env python3

#################################################################################
## Train neural net model for XRED (i.e. atomic positions)
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
def load_data(YdataType,saveDir,dataDir,trainSize):

    # Create directory for preprocessing results
    create_directory(saveDir)

    # Load data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)
    filename = dataDir + 'mg_RPRIM.npy'
    RPRIM = np.load(filename)

    trainX = X[:trainSize]
    trainY = Y[:trainSize]
    testX = X[2000:3000]
    testY = Y[2000:3000]
    testRPRIM = RPRIM[2000:3000]

    return trainX,testX,trainY,testY,testRPRIM

#--------------------------------------------------------------------------------
# Load raw data, perform PCA and rescale data
#--------------------------------------------------------------------------------
def preprocess_data(saveDir,trainX,trainY):

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
##-------------------------------------------------------------------------------
def build_model(nInDim):
    nHidden1 = 50
    nHidden2 = 100
    nHidden3 = 50
    model = keras.Sequential([
        Dense(nHidden1, activation='tanh', input_shape=[nInDim], kernel_regularizer=l2(0.0001)),
        Dense(nHidden2, activation='tanh', kernel_regularizer=l2(0.0001)),
        Dense(nHidden3, activation='tanh', kernel_regularizer=l2(0.0001)),
        Dense(3)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

    return model

##-------------------------------------------------------------------------------
## Evaluate ML test results
##-------------------------------------------------------------------------------
def evaluate_test_results(model,YdataType,saveDir,dataDir,testX,testY,testRPRIM,acell):

    # Load scalers
    scalerX = joblib.load(saveDir+'scalerX')
    scalerY = joblib.load(saveDir+'scalerY')

    # Evaluate and rescale
    originalTestX = np.copy(testX)
    testX = scalerX.transform(testX)
    predY = model.predict(testX)
    predY = scalerY.inverse_transform(predY)

    # Save some prediction results
    with open(saveDir+'prediction_sample.pickle' ,'wb') as f:
        pickle.dump([originalTestX[0],testY[0],predY[0]], f)

    # Compute relative error (RE), maximum absolute error (MAE),
    # displacement error (in Bohr) and true displacements (in Bohr)
    residue = predY - testY
    dataSize = np.shape(predY)[0]
    RSE = np.empty((dataSize))
    MAE = np.empty((dataSize))
    dispError = np.empty((dataSize))
    trueDisp =  np.empty((dataSize)) 
    for iData in range(dataSize):
        RSE[iData] = LA.norm(residue[iData])/LA.norm(testY[iData])
        MAE[iData] = np.amax(np.abs(residue[iData]))
        # Compute error in terms of displacement
        a = originalTestX[iData,0]
        b = originalTestX[iData,1]
        c = originalTestX[iData,2]
        rprim = testRPRIM[iData]
        dispError[iData] = LA.norm( residue[iData,0]*acell[0]*a*rprim[0:3] +
                                    residue[iData,1]*acell[1]*b*rprim[3:6] +
                                    residue[iData,2]*acell[2]*c*rprim[6:9] )
        trueDisp[iData] = LA.norm( testY[iData,0]*acell[0]*a*rprim[0:3] +
                                   testY[iData,1]*acell[1]*b*rprim[3:6] +
                                   testY[iData,2]*acell[2]*c*rprim[6:9] )

    meanRSE = np.mean(RSE)
    maxRSE = np.amax(RSE)
    meanMAE = np.mean(MAE)
    maxMAE = np.amax(MAE)
    meanDispError = np.mean(dispError)
    maxDispError = np.amax(dispError)
    meanTrueDisp = np.mean(trueDisp)
    maxTrueDisp = np.amax(trueDisp)
    print("Mean RSE: ",meanRSE)
    print("Max RSE: ",maxRSE)
    print("Mean MAE: ",meanMAE)
    print("Max MAE: ",maxMAE)
    print("Mean displacement error (Bohr): ",meanDispError)
    print("Max displacement error (Bohr): ",maxDispError)
    print("Mean true displacement (Bohr): ",meanTrueDisp)
    print("Max true displacement (Bohr): ",maxTrueDisp)
    np.savetxt(saveDir+'ml_test_rse.txt',RSE)
    np.savetxt(saveDir+'ml_test_mae.txt',MAE)
    np.savetxt(saveDir+'ml_test_disp_error.txt',dispError)
    np.savetxt(saveDir+'ml_test_true_disp.txt',trueDisp)

    # Save predY
    np.save(saveDir+'predY.npy',predY)

    return

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Set parameters
    YdataType = 'XRED'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
    trainSize = int(sys.argv[1])
    nInDim = 6
#    acell = [6.0419031073,10.464702002,9.8453372801]
    acell = [5.89,10.201779256580686,9.56]
    dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'
    saveDir = 'Saved_ml_results_occopt4_train'+str(trainSize)+'/'+YdataType+'/'

    # Time
    time0 = time.time()

    # Load raw data
    print("Loading data...")
    trainX,testX,trainY,testY,testRPRIM = load_data(YdataType,saveDir,dataDir,trainSize)

    # Preprocessing and perform ML
    trainNN = 1
    if trainNN==1:
        print("Preprocessing data...")
        trainX,trainY = preprocess_data(saveDir,trainX,trainY)
        print("Running ML...")
        model = build_model(nInDim)
        model.summary()
        history = model.fit(trainX,trainY,batch_size=128,epochs=4000,verbose=0)
        model.save(saveDir+'ml_model.h5')
        #model = LR().fit(trainX,trainY)
    else:
        print("Loading ML model...")
        model = keras.models.load_model(saveDir+'ml_model.h5')

    # Evaluate test errors (Note testY has not undergone preprocessing)
    evaluate_test_results(model,YdataType,saveDir,dataDir,testX,testY,testRPRIM,acell)

    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

if __name__ == '__main__': main()
