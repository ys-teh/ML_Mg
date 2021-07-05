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
from matplotlib.patches import ConnectionPatch
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
# Load trueXRED
#--------------------------------------------------------------------------------
def load_trueXRED():

    trueXRED = np.array([[6.8934131173E-17,8.3313463055E-02,5.0000000000E-01,5.8331346305E-01,\
                          6.7980671060E-17,5.8335320361E-01,5.0000000000E-01,8.3353203612E-02],\
                         [1.0619854010E-16,1.1497874808E-02,5.0000000000E-01,5.1149787481E-01,\
                          1.0648988497E-16,6.5516879186E-01,5.0000000000E-01,1.5516879186E-01],\
                         [-7.7239511183E-20,7.2827941255E-04,5.0000000000E-01,5.0072827941E-01,\
                          7.7239511183E-20,6.6593838725E-01,5.0000000000E-01,1.6593838725E-01],\
                         [4.7321003168E-17,-4.7058282448E-03,5.0000000000E-01,4.9529417176E-01,\
                          4.7281529648E-17,6.7137249491E-01,5.0000000000E-01,1.7137249491E-01],\
                         [2.8951917067E-17,-8.6622745983E-03,5.0000000000E-01,4.9133772540E-01,\
                          2.8953791905E-17,6.7532894126E-01,5.0000000000E-01,1.7532894126E-01]])
    return trueXRED

#--------------------------------------------------------------------------------
# Predict atomic positions given strain
#--------------------------------------------------------------------------------
def predict_XRED(strain,saveDir):

    # Load models
    scalerX = joblib.load(saveDir+'scalerX')
    scalerY = joblib.load(saveDir+'scalerY')
    model = keras.models.load_model(saveDir+'ml_model.h5')

    # Evaluate
    XRED = np.copy(strain)
    XRED = scalerX.transform(XRED)
    XRED = model.predict(XRED)
    XRED = scalerY.inverse_transform(XRED)

    return XRED

#--------------------------------------------------------------------------------
# Predict electron density given strain
#--------------------------------------------------------------------------------
def predict_DEN(strain,saveDir):

    # Load models
    scalerX = joblib.load(saveDir+'scalerX')
    scalerY = joblib.load(saveDir+'scalerY')
    pca = joblib.load(saveDir+'pca')
    model = keras.models.load_model(saveDir+'ml_model.h5')

    # Evaluate
    DEN = np.copy(strain)
    DEN = scalerX.transform(DEN)
    DEN = model.predict(DEN)
    DEN = scalerY.inverse_transform(DEN)
    DEN = pca.inverse_transform(DEN)

    return DEN

#--------------------------------------------------------------------------------
# Plot XRED and DEN
#--------------------------------------------------------------------------------
def plot_results(strain,XRED,DEN,trueXRED,acell,ngfft):

    x1coord = np.empty((ngfft[0]+1,ngfft[1]+1))
    x2coord = np.empty((ngfft[0]+1,ngfft[1]+1))
    Y = np.empty((ngfft[0]+1,ngfft[1]+1))

#    vertices = np.array([[-acell[0]/2.,-acell[1]/2.],\
#                         [acell[0]/2.,-acell[1]/2.],\
#                         [acell[0]/2.,acell[1]/2.],\
#                         [-acell[0]/2.,acell[1]/2.],\
#                         [-acell[0]/2.,-acell[1]/2.]])

    istrain = -1
    ishift = 0
    fig,axes = plt.subplots(nrows=2,ncols=5,figsize=(10,5.2))
    plt.subplots_adjust(wspace=0.1,hspace=0.02)
    for ax in axes.flat:
        istrain += 1
        if istrain>=5 :
            istrain = 0
            ishift = int(ngfft[0]*ngfft[1]*ngfft[2]/2)
        a = strain[istrain,0]
        b = strain[istrain,1]
        c = strain[istrain,2]
        r = strain[istrain,3]
        s = strain[istrain,4]
        t = strain[istrain,5]
        cos_r = math.cos(math.radians(r))
        cos_s = math.cos(math.radians(s))
        cos_t = math.cos(math.radians(t))
        dV = (acell[0]*acell[1]*acell[2]*a*b*c)/(ngfft[0]*ngfft[1]*ngfft[2]) \
             * math.sqrt( 1 + 2*cos_r*cos_s*cos_t - cos_r*cos_r - cos_s*cos_s - cos_t*cos_t ) 

        # Plot DEN at z=0 or z=1/2
        for i1 in range(ngfft[0]+1):
            for i2 in range(ngfft[1]+1):
                x1coord[i1,i2] = float(i1)/ngfft[0]*acell[0]*a + float(i2)/ngfft[1]*acell[1]*b*cos_t
                x1coord[i1,i2] -= acell[0]*a/2.
                x2coord[i1,i2] = float(i2)/ngfft[1]*acell[1]*b*math.sin(math.radians(t))
                x2coord[i1,i2] -= acell[1]*b/2.
                if i1 is ngfft[0]:
                    n1 = 0
                else:
                    n1 = i1
                if i2 is ngfft[1]:
                    n2 = 0
                else:
                    n2 = i2
                Y[i1,i2] = DEN[istrain, ishift+n1+n2*ngfft[0] ]/dV*1000.0
        im = ax.contourf(x1coord,x2coord,Y,50,vmin=3,vmax=16)
        ax.set_xlim(-acell[0]*1.2/2.,acell[0]*1.2/2.)
        ax.set_ylim(-acell[1]*1.2/2.,acell[1]*1.2/2.)
#        ax.plot(vertices[:,0],vertices[:,1],'k--',linewidth=0.5)
        if ishift is 0:
            vertices1 = np.array([[-acell[0]*a/2.,0],\
                         [acell[0]*a/2.,0]])
            ax.plot(vertices1[:,0],vertices1[:,1],'k--',linewidth=0.5)
        else:
            vertices1 = np.array([[-acell[0]*a/2.,acell[1]*b/6.],\
                         [acell[0]*a/2.,acell[1]*b/6.]])
            ax.plot(vertices1[:,0],vertices1[:,1],'k--',linewidth=0.5)
            vertices2 = np.array([[-acell[0]*a/2.,-acell[1]*b/3.],\
                         [acell[0]*a/2.,-acell[1]*b/3.]])
            ax.plot(vertices2[:,0],vertices2[:,1],'k--',linewidth=0.5)

        ax.axis("off")

        # Plot XRED at z=0 and z=1/2
        atomPos = np.zeros((2,2))
        tAtomPos = np.zeros((2,2))
        if ishift is 0:
            atomPos[0,0] = (XRED[istrain,0]-0.5)*acell[0]*a
            atomPos[0,1] = (XRED[istrain,1]-0.5)*acell[1]*b
            atomPos[1,0] = (XRED[istrain,0])*acell[0]*a
            atomPos[1,1] = (XRED[istrain,1])*acell[1]*b
            tAtomPos[0,0] = (trueXRED[istrain,0]-0.5)*acell[0]*a
            tAtomPos[0,1] = (trueXRED[istrain,1]-0.5)*acell[1]*b
            tAtomPos[1,0] = (trueXRED[istrain,2]-0.5)*acell[0]*a
            tAtomPos[1,1] = (trueXRED[istrain,3]-0.5)*acell[1]*b
        else:
            atomPos[0,0] = (-XRED[istrain,0]-0.5)*acell[0]*a
            atomPos[0,1] = (1./6.-XRED[istrain,1])*acell[1]*b
            atomPos[1,0] = (-XRED[istrain,0])*acell[0]*a
            atomPos[1,1] = (-1./3.-XRED[istrain,1])*acell[1]*b
            tAtomPos[0,0] = (trueXRED[istrain,4]-0.5)*acell[0]*a
            tAtomPos[0,1] = (trueXRED[istrain,5]-0.5)*acell[1]*b
            tAtomPos[1,0] = (trueXRED[istrain,6]-0.5)*acell[0]*a
            tAtomPos[1,1] = (trueXRED[istrain,7]-0.5)*acell[1]*b

        ax.scatter(tAtomPos[:,0],tAtomPos[:,1],color='pink',marker='+',clip_on=False)
        ax.scatter(atomPos[:,0],atomPos[:,1],color='red',marker='x',clip_on=False)

        # Title
        if ishift is 0:
            ax.set_title(r'$\varepsilon_{22} = $ %g' %(strain[istrain,1]-1.0))

    fig.colorbar(im,ax=axes.ravel().tolist())
    plt.figtext(0.03,0.7,r'$\overline{x}_3=0$')
    plt.figtext(0.03,0.27,r'$\overline{x}_3=1/2$')
    plt.figtext(0.78,0.9,r'$10^{-3}$ $e$ Bohr$^{-3}$')
    plt.figtext(0.046,0.49,r'$\uparrow$')
    plt.figtext(0.05,0.47,r'$\rightarrow$')
    plt.figtext(0.08,0.46,r'$x_1$')
    plt.figtext(0.043,0.53,r'$x_2$')
    plt.suptitle('(a)',y=1.01)
   
    plt.savefig('Figures/visualize_uniaxial_test.svg')
#    plt.show()


    return

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

    # Load strain values
    strain = load_strain()

    # Load true XRED
    trueXRED = load_trueXRED()

    # Predict XRED
    XRED = predict_XRED(strain,saveDirXRED)

    # Predict DEN
    DEN = predict_DEN(strain,saveDirDEN)

    # Plot results
    plot_results(strain,XRED,DEN,trueXRED,acell,ngfft)


if __name__ == '__main__': main()
