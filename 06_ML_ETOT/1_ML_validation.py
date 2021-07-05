#/usr/bin/env python3

#################################################################################
# k-fold cross validation for neural net model for ETOT (i.e. total free energy)
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
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import GradientBoostingRegressor as GBR
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
def load_data(YdataType,dataDir):

    # Load data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)

    X = X[:2000]
    Y = Y[:2000]

    return X,Y

#--------------------------------------------------------------------------------
# Load raw data and rescale data
#--------------------------------------------------------------------------------
def preprocess_data(trainX,testX,trainY,testY):

    # Rescale data
    scalerX = StandardScaler()
    trainX = scalerX.fit_transform(trainX)
    testX = scalerX.transform(testX)
    scalerY = StandardScaler()
    trainY = scalerY.fit_transform(trainY)
    testY = scalerY.transform(testY)

    return trainX,testX,trainY,testY,scalerX,scalerY

##-------------------------------------------------------------------------------
## Machine learning model
## nInDim = input dimension of model
##-------------------------------------------------------------------------------
def build_model(nInDim):
    nHidden1 = 50
    nHidden2 = 50
    nHidden3 = 50
#    nHidden4 = 50
    model = keras.Sequential([
        Dense(nHidden1, activation='tanh', input_shape=[nInDim], kernel_regularizer=l2(0.0001)),
        Dense(nHidden2, activation='tanh', kernel_regularizer=l2(0.0001)),
        Dense(nHidden3, activation='tanh', kernel_regularizer=l2(0.0001)),
#        Dense(nHidden4, activation='sigmoid', kernel_regularizer=l2(0.0001)),
        Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

    return model

##-------------------------------------------------------------------------------
## Evaluate ML test results
##-------------------------------------------------------------------------------
def evaluate_test_results(model,YdataType,dataDir,testID,scalerX,scalerY,acell):

    # Load test data
    filename = dataDir + 'mg_Y_' + YdataType + '.npy'
    Y = np.load(filename)
    filename = dataDir + 'mg_X.npy'
    X = np.load(filename)
    testX = X[testID]
    testY = Y[testID]

    # Evaluate and rescale
    originalTestX = np.copy(testX)
    testX = scalerX.transform(testX)
    predY = model.predict(testX)
    predY = scalerY.inverse_transform(predY)

    # Compute normalized squared error
    residue = np.absolute(predY - testY)
    relError = residue/np.absolute(testY)
    maxAbsError = np.max(residue)
    meanAbsError = np.mean(residue)
    maxRelError = np.max(relError)
    meanRelError = np.mean(relError)
    print("Max absolute error: ",maxAbsError)
    print("Mean absolute error: ",meanAbsError)
    print("Max relative error: ",maxRelError)
    print("Mean relative error: ",meanRelError)

    return maxAbsError,meanAbsError,maxRelError,meanRelError

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Set parameters
    YdataType = 'ETOT'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
    acell = [5.89,10.201779256580686,9.56]
    nInDim = 6
    dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'

    # Time
    time0 = time.time()

    # Load raw data
    print("Loading data...")
    X,Y = load_data(YdataType,dataDir)

    # Perform k-fold cross-validation
    k = 4
    allErrors = np.empty((k,4))
    kf = KFold(n_splits=k)
    fold = 0
    for trainID, testID in kf.split(X):
#        if fold == 3:
#            break
        trainX, testX = X[trainID], X[testID]
        trainY, testY = Y[trainID], Y[testID]
        trainX,testX,trainY,testY,scalerX,scalerY = preprocess_data(trainX,testX,trainY,testY)

        # Perform ML
        print("Running ML...")
        model = build_model(nInDim)
        model.summary()
        #callback = keras.callbacks.EarlyStopping(monitor='loss',patience=5,verbose=1,min_delta=0.0)
        history = model.fit(trainX,trainY,batch_size=128,epochs=4000,verbose=0,validation_data=(testX,testY)) #,callbacks=[callback])
        #model = RFR(max_depth=10,min_samples_split=2,n_estimators=1000,min_samples_leaf=1,random_state=0)
        #model = ABR(n_estimators=50,learning_rate=1,loss='square',random_state=0)
        #model = DTR(max_depth=5,min_samples_split=30,min_samples_leaf=20)
        #model = GBR(learning_rate=0.1,n_estimators=100,subsample=1.0,min_samples_split=2,min_samples_leaf=1,max_depth=3)
        #model.fit(trainX,trainY)
    
        # Evaluate test errors
        maxAbsError,meanAbsError,maxRelError,meanRelError = evaluate_test_results(model,YdataType,dataDir,testID,scalerX,scalerY,acell)
        allErrors[fold] = [maxAbsError,meanAbsError,maxRelError,meanRelError]
#        print("Loss: ",history.history['loss'][-1])
#        print("Val loss: ",history.history['val_loss'][-1])

        fold += 1

    # Print average errors
    print("\n")
    print("Averaging results from all folds...")
    print("Max absolute error: ",np.mean(allErrors[:,0]))
    print("Mean absolute error: ",np.mean(allErrors[:,1]))
    print("Max relative error: ",np.mean(allErrors[:,2]))
    print("Mean relative error: ",np.mean(allErrors[:,3]))

    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

#    # Plot ML results
#    print("Plotting results...")
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.xlabel('epoch')
#    plt.ylabel('loss')
#    plt.legend(['train','test'], loc='upper left')
#    plt.show()

if __name__ == '__main__': main()
