#/usr/bin/env python3

#################################################################################
# Generate stress strain curve from ML energy values
# Compare with Abinit results
# Create by Ying Shi Teh
# Last modified on July 5, 2021
#################################################################################

import numpy as np
from numpy import linalg as LA
import math
import os
import joblib
import time
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
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})
from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------
# Import models
#-------------------------------------------------------------------------------
def import_models(saveDir):

    # Import NN model
    model = keras.models.load_model(saveDir+'ml_model.h5')
    weights = model.get_weights()

    # Import scaler
    scalerX = joblib.load(saveDir+'scalerX')
    scalerX_mean = scalerX.mean_
    scalerX_scale = scalerX.scale_
    scalerY = joblib.load(saveDir+'scalerY')
    scalerY_mean = scalerY.mean_[0]
    scalerY_scale = scalerY.scale_[0]

    return weights,scalerX_mean,scalerX_scale,scalerY_mean,scalerY_scale

#-------------------------------------------------------------------------------
# Compute energy value and its gradient
# Input: x[1,6]
# Output: 
#-------------------------------------------------------------------------------
def compute_value_and_gradient(x,weights,scalerX_mean,scalerX_scale,\
           scalerY_mean,scalerY_scale):

    # Compute energy value
    nHidden = 3
    #z = [None]*nHidden
    #derivSigmoid = [None]*nHidden
    derivTanh = [None]*nHidden
    y = (x - scalerX_mean)/scalerX_scale
    for h in range(nHidden):
        y = np.matmul(y,weights[h*2])
        y += weights[h*2+1]
        #z[h] = (y>=0)
        #y *= (y>0)
        #derivSigmoid[h] = np.exp(-y)/np.square(1+np.exp(-y))
        #y = 1/(1+np.exp(-y))
        derivTanh[h] = 1 - np.square(np.tanh(y))
        y = np.tanh(y)
    y = np.matmul(y,weights[nHidden*2])
    y += weights[nHidden*2+1]
    y = y*scalerY_scale + scalerY_mean

    for i in range(nHidden):
        #z[i] = np.transpose(z[i])
        #derivSigmoid[i] = np.transpose(derivSigmoid[i])
        derivTanh[i] = np.transpose(derivTanh[i])

    # Compute gradient
    grady = weights[nHidden*2]*scalerY_scale
    for h in range(nHidden-1,-1,-1):
        grady = np.matmul(weights[h*2],np.multiply(derivTanh[h],grady))
    grady = np.multiply(np.reshape(grady,(1,6)),1./scalerX_scale)

    return y,grady

#-------------------------------------------------------------------------------
# Gradient descent (BB method)
# Input: x[1,6], relax[1,6]
#-------------------------------------------------------------------------------
def gradient_descent(x,relax,weights,scalerX_mean,scalerX_scale,\
           scalerY_mean,scalerY_scale):

    nIterations = 2000  # Maximum number of iterations
    gamma = 0.5  # Step size
    tol = 1e-6   # Tolerance
    for iter in range(nIterations):
        print('Iteration: ',iter)
        y,grady = compute_value_and_gradient(x,weights,scalerX_mean,scalerX_scale,\
           scalerY_mean,scalerY_scale)

        # Stopping criterion
        if np.max(np.absolute(relax*grady)) < tol:
            print('Convergence has reached at iteration ',iter)
            break
        elif (iter == (nIterations-1)):
            print('Solution is not converged!')

        # Standard gradient descent
        if iter is 0:
            gradyOld = np.copy(grady)
        # BB method
        else:
            gradyDiff = relax*(grady - gradyOld)
            denominator = np.sum(np.square(gradyDiff))
            if denominator > 0:
                gamma = np.absolute(np.sum(np.multiply(deltax,gradyDiff)))/denominator
            gradyOld = np.copy(grady)

        # Update x
        deltax = -gamma*relax*grady
        x += deltax
        print('Gradient: ',grady)
        print('Strain: ',x)

    return x,grady

#-------------------------------------------------------------------------------
# Stress-strain
# Input: x[1,6], relax[1,6]
#-------------------------------------------------------------------------------
def plot_stress_strain(stressDirection,weights,scalerX_mean,scalerX_scale,scalerY_mean,scalerY_scale):

    N = 61
    strain = np.linspace(-0.1,0.1,N)
    stress = np.zeros((N))
    allStrain = np.zeros((N,6))

    x = np.array([[1.0,1.0,1.0,90.0,90.0,90.0]])
    relax = np.array([[1.,1.,1.,0.,0.,0.]])
    relax[0,stressDirection] = 0.
    for i in range(math.floor(N/2),N):
        x[0,stressDirection] = 1.0 + strain[i]
        x,grady = gradient_descent(x,relax,weights,scalerX_mean,scalerX_scale,\
           scalerY_mean,scalerY_scale)
        vol = acell[0]*acell[1]*acell[2]*x[0,0]*x[0,1]*x[0,2]
        stress[i] = grady[0,stressDirection]/vol
        allStrain[i] = x
    x = np.array([[1.0,1.0,1.0,90.0,90.0,90.0]])
    for i in range(math.floor(N/2)-1,-1,-1):
        x[0,stressDirection] = 1.0 + strain[i]
        x,grady = gradient_descent(x,relax,weights,scalerX_mean,scalerX_scale,\
           scalerY_mean,scalerY_scale)
        vol = acell[0]*acell[1]*acell[2]*x[0,0]*x[0,1]*x[0,2]
        stress[i] = grady[0,stressDirection]/vol
        allStrain[i] = x
    np.save('Uniaxial_test_results/strain_direction'+str(stressDirection+1)+'.npy',allStrain)

    # Load abinit stress values
    strainAbinit = np.linspace(-0.1,0.1,21)
    stressAbinit = np.loadtxt('Abinit_runs/direction'+str(stressDirection+1)+'/mg_summarizedStressValues.txt')
    residualStress = stressAbinit[10]

    markerType = ['o','x','^']
    label1 = [r'ML ($x_1$)',r'ML ($x_2$)',r'ML ($x_3$)']
    label2 = [r'True ($x_1$)',r'True ($x_2$)',r'True ($x_3$)']
    locations = ['lower center','center right','lower right']
    if stressDirection is 0:
        plt.figure(figsize=(5,3.5))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.plot(strain,(stress-residualStress)*1000,label=label1[stressDirection],color='C'+str(stressDirection))
    plt.plot(strainAbinit,(stressAbinit-residualStress)*1000,markerType[stressDirection],label=label2[stressDirection],color='C'+str(stressDirection),mfc='none')

#    p1, = plt.plot(strain,(stress-residualStress)*1000,label=label1[stressDirection],color='C'+str(stressDirection))
#    p2, = plt.plot(strainAbinit,(stressAbinit-residualStress)*1000,markerType[stressDirection],label=label2[stressDirection],color='C'+str(stressDirection),mfc='none')
#
#    if stressDirection is 0:
#        legend1 = plt.legend(handles=[p1,p2],frameon=False,labelspacing=0.18,loc=locations[stressDirection],fontsize='medium',handletextpad=0.4,handlelength=0.8,bbox_to_anchor=(0.4, 0.0))
#    else:
#        legend1 = plt.legend(handles=[p1,p2],frameon=False,labelspacing=0.18,loc=locations[stressDirection],fontsize='medium',handletextpad=0.4,handlelength=0.8)
#    ax = plt.gca().add_artist(legend1)
    if stressDirection is 2:
        plt.xlabel('Strain')
        plt.ylabel('Stress (mHa/Bohr$^3$)')
        plt.legend(frameon=False,labelspacing=0.18,loc='lower right',handletextpad=0.4)
        plt.title('(a)',fontsize=16)
        plt.savefig('Figures/stress_strain.svg', bbox_inches='tight')
        #plt.show()

    return

#-------------------------------------------------------------------------------
# Stress-strain
# Input: x[1,6], relax[1,6]
#-------------------------------------------------------------------------------
def plot_transverse_strain(direction):

    N = 61
    strain = np.linspace(-0.1,0.1,N)
    strainAbinit = np.linspace(-0.1,0.1,21)
    markerType = ['o','x','^']
    locations = ['upper center','upper right']

    Ddirect = np.load('Uniaxial_test_results/strain_direction'+str(direction+1)+'.npy')
    Ddirect -= 1.0
    Dtrue = np.loadtxt('Abinit_runs/direction'+str(direction+1)+'/mg_summarizedStrainValues.txt')
    allStrainDirections = [0,1,2]
    allStrainDirections.remove(direction)
    allTitles = ['(b)','(c)','(d)']

    plt.figure(figsize=(5,3.5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for strainDirection in allStrainDirections:
        plt.plot(strain,Ddirect[:,strainDirection]-Dtrue[10,strainDirection],label=r'ML $\varepsilon_{'+str(strainDirection+1)+str(strainDirection+1)+'}$',color='C'+str(strainDirection))
        plt.plot(strainAbinit,Dtrue[:,strainDirection]-Dtrue[10,strainDirection],markerType[strainDirection],label=r'True $\varepsilon_{'+str(strainDirection+1)+str(strainDirection+1)+'}$',color='C'+str(strainDirection),mfc='none')

    plt.xlabel(r'Strain($\varepsilon_{'+str(direction+1)+str(direction+1)+'}$)')
    plt.ylabel('Transverse strain')
    plt.legend(frameon=False,labelspacing=0.18,loc='upper right',handletextpad=0.4)
    plt.title(allTitles[direction],fontsize=16)
    plt.savefig('Figures/transverse_strain'+str(direction+1)+'.svg', bbox_inches='tight')

    return

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
# Set parameters
YdataType = 'ETOT'        # data type ('DEN' or 'VCLMB' or 'ENTR' or 'EBAND')
trainSize = 2000
nInDim = 6
acell = [5.89,10.201779256580686,9.56]
dataDir = '../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/'
saveDir = 'Saved_ml_results_occopt4_train'+str(trainSize)+'/'+YdataType+'/'

def main():

    weights,scalerX_mean,scalerX_scale,scalerY_mean,scalerY_scale = import_models(saveDir)

    stressDirection = 0
    plot_stress_strain(stressDirection,weights,scalerX_mean,scalerX_scale,scalerY_mean,scalerY_scale)

    stressDirection = 1
    plot_stress_strain(stressDirection,weights,scalerX_mean,scalerX_scale,scalerY_mean,scalerY_scale)

    stressDirection = 2
    plot_stress_strain(stressDirection,weights,scalerX_mean,scalerX_scale,scalerY_mean,scalerY_scale)

    # Plot strains in orthogonal directions
    direction = 0
    plot_transverse_strain(direction)
    direction = 1
    plot_transverse_strain(direction)
    direction = 2
    plot_transverse_strain(direction)


if __name__ == '__main__': main()
