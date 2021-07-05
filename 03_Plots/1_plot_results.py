#/usr/bin/env python3

#################################################################################
## Plots for paper
## Created by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
from numpy import linalg as LA
import math
import os
import joblib
import time
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})
from matplotlib import pyplot as plt
matplotlib.pyplot.subplots
matplotlib.axis.Axis.set_major_formatter
matplotlib.axis.Axis.set_major_locator
matplotlib.axis.Axis.set_minor_locator
matplotlib.ticker.AutoMinorLocator
matplotlib.ticker.MultipleLocator
matplotlib.ticker.StrMethodFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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

##-------------------------------------------------------------------------------
## Compute volumetric strain for testX
##-------------------------------------------------------------------------------
def compute_vol_strain_for_testX(testX):

    dataSize = np.shape(testX)[0]
    volStrain = np.empty((dataSize))
    for iData in range(dataSize):
        a = testX[iData,0]
        b = testX[iData,1]
        c = testX[iData,2]
        r = testX[iData,3]
        s = testX[iData,4]
        t = testX[iData,5]
        cos_r = math.cos(math.radians(r))
        cos_s = math.cos(math.radians(s))
        cos_t = math.cos(math.radians(t))
        volStrain[iData] = (a*b*c) \
         * math.sqrt( 1 + 2*cos_r*cos_s*cos_t - cos_r*cos_r - cos_s*cos_s - cos_t*cos_t )

    volStrain -= 1.

    return volStrain

##-------------------------------------------------------------------------------
## Plot error vs volumetric strain
##-------------------------------------------------------------------------------
def plot_error_vs_vol_strain(saveDir,volStrain):

    # DEN
    YdataType = 'DEN'
    saveDirY = saveDir + YdataType + '/'
    fig,ax = plt.subplots(1,4,figsize=(12,2.5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[0].scatter(volStrain,relError,label='NN',alpha=0.5,clip_on=False)#,facecolors='none',edgecolors='k')
    ax[0].set_title('Electron density',fontsize=16)
    ax[0].set_xlabel('Volumetric strain')
    ax[0].set_ylabel('NRMSE')
    ax[0].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].set_xlim((-0.3,0.3))
    ax[0].set_ylim((0.0,0.08))

    # VCLMB
    YdataType = 'VCLMB'
    saveDirY = saveDir + YdataType + '/'
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[1].scatter(volStrain,relError,label='NN',alpha=0.5,clip_on=False)
    ax[1].set_title('Coulomb potential',fontsize=16)
    ax[1].set_xlabel('Volumetric strain')
    ax[1].set_ylim((0.0,0.08))
    ax[1].set_xlim((-0.3,0.3))
    ax[1].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))

    # EBAND
    YdataType = 'EBAND'
    saveDirY = saveDir + YdataType + '/'
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[2].scatter(volStrain,relError,label='NN',alpha=0.5)
    ax[2].set_title(r'\begin{center} Band structure \\ energy density \end{center}',fontsize=16)
    ax[2].set_xlabel('Volumetric strain')
    ax[2].set_ylim((0.0,0.08))
    ax[2].set_xlim((-0.3,0.3))
    ax[2].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[2].xaxis.set_minor_locator(MultipleLocator(0.1))

    # ENTR
    YdataType = 'ENTR'
    saveDirY = saveDir + YdataType + '/'
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[3].scatter(volStrain,relError,label='NN',alpha=0.5)
    ax[3].set_title('Volumetric entropy',fontsize=16)
    ax[3].set_xlabel('Volumetric strain')
    ax[3].set_ylim((0.0,1.5))
    ax[3].set_xlim((-0.3,0.3))
    ax[3].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[3].xaxis.set_minor_locator(MultipleLocator(0.1))


    fig.savefig('Figures/error_vs_vol_strain.pdf', bbox_inches='tight')

    return
##-------------------------------------------------------------------------------
## Plot error vs volumetric strain (version 2)
##-------------------------------------------------------------------------------
def plot_error_vs_vol_strain2(saveDir,volStrain):

    # DEN
    YdataType = 'DEN'
    saveDirY = saveDir + YdataType + '/'
    fig,ax = plt.subplots(1,4,figsize=(12,2.5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    relError = np.loadtxt(saveDirY+'linear_test_nrmse.txt')
    ax[0].scatter(volStrain,relError,marker='d',label='LR',alpha=0.5,color='C1')
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[0].scatter(volStrain,relError,label='NN',alpha=0.5,color='C0')
    ax[0].set_title('Electron density',fontsize=16)
    ax[0].set_xlabel('Volumetric strain')
    ax[0].set_ylabel('NRMSE')
    ax[0].set_ylim((0.0,0.15))
    ax[0].set_xlim((-0.3,0.3))
    ax[0].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].legend(loc='upper right',bbox_to_anchor=(1.05,1.05),handletextpad=-0.3,frameon=False)

    # VCLMB
    YdataType = 'VCLMB'
    saveDirY = saveDir + YdataType + '/'
    relError = np.loadtxt(saveDirY+'linear_test_nrmse.txt')
    ax[1].scatter(volStrain,relError,marker='d',label='LR',alpha=0.5,color='C1')
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[1].scatter(volStrain,relError,label='NN',alpha=0.5,color='C0')
    ax[1].set_title('Coulomb potential',fontsize=16)
    ax[1].set_xlabel('Volumetric strain')
    ax[1].set_ylim((0.0,0.15))
    ax[1].set_xlim((-0.3,0.3))
    ax[1].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))

    # EBAND
    YdataType = 'EBAND'
    saveDirY = saveDir + YdataType + '/'
    relError = np.loadtxt(saveDirY+'linear_test_nrmse.txt')
    ax[2].scatter(volStrain,relError,marker='d',label='LR',alpha=0.5,color='C1')
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[2].scatter(volStrain,relError,label='NN',alpha=0.5,color='C0')
    ax[2].set_xlabel('Volumetric strain')
    ax[2].set_title(r'\begin{center} Band structure \\ energy density \end{center}',fontsize=16)
    ax[2].set_ylim((0.0,0.10))
    ax[2].set_xlim((-0.3,0.3))
    ax[2].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[2].xaxis.set_minor_locator(MultipleLocator(0.1))


    # ENTR
    YdataType = 'ENTR'
    saveDirY = saveDir + YdataType + '/'
    relError = np.loadtxt(saveDirY+'linear_test_nrmse.txt')
    ax[3].scatter(volStrain,relError,marker='d',label='LR',alpha=0.5,color='C1')
    relError = np.loadtxt(saveDirY+'ml_test_nrmse.txt')
    ax[3].scatter(volStrain,relError,label='NN',alpha=0.5,color='C0')
    ax[3].set_title('Volumetric entropy',fontsize=16)
    ax[3].set_xlabel('Volumetric strain')
    ax[3].set_ylim((0.0,1.25))
    ax[3].set_xlim((-0.3,0.3))
    ax[3].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[3].xaxis.set_minor_locator(MultipleLocator(0.1))

    fig.savefig('Figures/error_vs_vol_strain2.pdf', bbox_inches='tight')

    return

##-------------------------------------------------------------------------------
## Plot error vs training size
##-------------------------------------------------------------------------------
def plot_error_vs_train_size(saveDir):

    saveMLfile = 'ml_test_nrmse.txt'
    savePCAfile = 'pca_test_nrmse.txt'
    saveDir = saveDir.split('/')
    saveDir = saveDir[0] + '/' + saveDir[1] + '/'
    trainSize = np.linspace(0,2000,9)
    trainSize = trainSize[1:]

    fig = plt.figure(figsize=(12,2.5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # DEN
    YdataType = 'DEN'
    meanError = []
    meanErrorPCA = []
    for i in range(np.shape(trainSize)[0]):
        saveDirY = saveDir + 'Saved_ml_results_occopt4_train' + str(int(trainSize[i])) + '/' + YdataType + '/'
        error = np.loadtxt(saveDirY + saveMLfile)
        meanError.append(np.mean(error))
        error = np.loadtxt(saveDirY + savePCAfile)
        meanErrorPCA.append(np.mean(error))

    sub = fig.add_subplot(141)
    sub.plot(trainSize,meanError,'kx--',clip_on=False)
    sub.plot(trainSize, meanErrorPCA,'r^--',clip_on=False)
    sub.set_xlim((0,2000))
    sub.set_ylim((0.0,0.006))
    #sub.set_yticks([0.000,0.001,0.002,0.003])
    sub.set_title('Electron density',fontsize=16)
    sub.set_xlabel('Training size')
    sub.set_ylabel('Mean NRMSE')
    sub.legend(['ML','PCA'],frameon=False)

    # VCLMB
    YdataType = 'VCLMB'
    meanError = []
    meanErrorPCA = []
    for i in range(np.shape(trainSize)[0]):
        saveDirY = saveDir + 'Saved_ml_results_occopt4_train' + str(int(trainSize[i])) + '/' + YdataType + '/'
        error = np.loadtxt(saveDirY + saveMLfile)
        meanError.append(np.mean(error))
        error = np.loadtxt(saveDirY + savePCAfile)
        meanErrorPCA.append(np.mean(error))

    sub = fig.add_subplot(142)
    sub.plot(trainSize,meanError,'kx--',clip_on=False)
    sub.plot(trainSize, meanErrorPCA,'r^--',clip_on=False)
    sub.set_xlim((0,2000))
    sub.set_ylim((0.0,0.006))
    #sub.set_yticks([0.000,0.001,0.002,0.003])
    sub.set_title('Coulomb potential',fontsize=16)
    sub.set_xlabel('Training size')
#    sub.set_ylabel('Mean NRMSE')


    # EBAND
    YdataType = 'EBAND'
    meanError = []
    meanErrorPCA = []
    for i in range(np.shape(trainSize)[0]):
        saveDirY = saveDir + 'Saved_ml_results_occopt4_train' + str(int(trainSize[i])) + '/' + YdataType + '/'
        error = np.loadtxt(saveDirY + saveMLfile)
        meanError.append(np.mean(error))
        error = np.loadtxt(saveDirY + savePCAfile)
        meanErrorPCA.append(np.mean(error))

    sub = fig.add_subplot(143)
    sub.plot(trainSize,meanError,'kx--',clip_on=False)
    sub.plot(trainSize, meanErrorPCA,'r^--',clip_on=False)
    sub.set_xlim((0,2000))
    sub.set_ylim((0.0,0.006))
    #sub.set_yticks([0.000,0.001,0.002,0.003])
    sub.set_title(r'\begin{center} Band structure \\ energy density \end{center}',fontsize=16)
    sub.set_xlabel('Training size')
#    sub.set_ylabel('Mean NRMSE')

    # ENTR
    YdataType = 'ENTR'
    meanError = []
    meanErrorPCA = []
    for i in range(np.shape(trainSize)[0]):
        saveDirY = saveDir + 'Saved_ml_results_occopt4_train' + str(int(trainSize[i])) + '/' + YdataType + '/'
        error = np.loadtxt(saveDirY + saveMLfile)
        meanError.append(np.mean(error))
        error = np.loadtxt(saveDirY + savePCAfile)
        meanErrorPCA.append(np.mean(error))

    sub = fig.add_subplot(144)
    sub.plot(trainSize,meanError,'kx--',clip_on=False)
    sub.plot(trainSize, meanErrorPCA,'r^--',clip_on=False)
    sub.set_xlim((0,2000))
    sub.set_ylim((0.0,0.4))
    #sub.set_yticks([0.000,0.001,0.002,0.003])
    sub.set_title('Volumetric entropy',fontsize=16)
    sub.set_xlabel('Training size')
#    sub.set_ylabel('Mean NRMSE')

    fig.savefig('Figures/error_vs_train_size.pdf', bbox_inches='tight')

    return

##-------------------------------------------------------------------------------
## Show one example of prediction for all variables
##-------------------------------------------------------------------------------
def show_one_prediction_for_all_variables(saveDir,acell,ngfft):

    # Load DEN
    with open(saveDir+'DEN/prediction_sample.pickle', 'rb') as f:
        [testX, testY, predY] = pickle.load(f)

    # Compute coordinates
    a = testX[0]
    b = testX[1]
    c = testX[2]
    r = testX[3]
    s = testX[4]
    t = testX[5]
    cos_r = math.cos(math.radians(r))
    cos_s = math.cos(math.radians(s))
    cos_t = math.cos(math.radians(t))
    x1coord = np.empty((ngfft[0],ngfft[1]))
    x2coord = np.empty((ngfft[0],ngfft[1]))
    predY2d = np.empty((ngfft[0],ngfft[1]))
    testY2d = np.empty((ngfft[0],ngfft[1]))

    # Compute volume of each 3D grid
    dV = (acell[0]*acell[1]*acell[2]*a*b*c)/(ngfft[0]*ngfft[1]*ngfft[2]) \
             * math.sqrt( 1 + 2*cos_r*cos_s*cos_t - cos_r*cos_r - cos_s*cos_s - cos_t*cos_t )

    ishift = 0
    for i1 in range(ngfft[0]):
        for i2 in range(ngfft[1]):
            x1coord[i1,i2] = float(i1)/ngfft[0]*acell[0]*a + float(i2)/ngfft[1]*acell[1]*b*cos_t
            x2coord[i1,i2] = float(i2)/ngfft[1]*acell[1]*b*math.sin(math.radians(t))
            predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]/dV
            testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]/dV
    testY2d *= 1000.0
    predY2d *= 1000.0

    fig = plt.figure(figsize=(11,10))
    plt.subplots_adjust(wspace=0.9, hspace=0.2)

    plt.figtext(0.04,0.65,'Ground truth',rotation=90)
    plt.figtext(0.04,0.45,'ML prediction',rotation=90)
    plt.figtext(0.04,0.2,'Error',rotation=90)

    sub = fig.add_subplot(4,4,5)
    CS = sub.contourf(x1coord,x2coord,testY2d,50) #vmin=0.0035,vmax=0.017
    sub.set_title(r'\begin{center} (a) \\ Electron density \\ ($10^{-3}$ a.u.) \end{center}',fontsize=16)
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS,ticks=[5,10,15,20])

    sub = fig.add_subplot(4,4,9)
    CS = sub.contourf(x1coord,x2coord,predY2d,50)
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS,ticks=[5,10,15,20])

    sub = fig.add_subplot(4,4,13)
    CS = sub.contourf(x1coord,x2coord,predY2d-testY2d,50,cmap='seismic')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS,ticks=[-2e-2,0,2e-2,4e-2])

    # Load VCLMB
    with open(saveDir+'VCLMB/prediction_sample.pickle', 'rb') as f:
        [testX, testY, predY] = pickle.load(f)

    for i1 in range(ngfft[0]):
        for i2 in range(ngfft[1]):
            predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]
            testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]
    predY2d *= 1000.0
    testY2d *= 1000.0

    sub = fig.add_subplot(4,4,6)
    CS = sub.contourf(x1coord,x2coord,testY2d,50) #vmin=0.0035,vmax=0.017
    cbar = plt.colorbar(CS,ticks=[-60,-30,0,30,60])
    sub.set_title(r'\begin{center} (b) \\ Coulomb potential \\ ($10^{-3}$ a.u.) \end{center}',fontsize=16,multialignment='center')

    sub = fig.add_subplot(4,4,10)
    CS = sub.contourf(x1coord,x2coord,predY2d,50)
    cbar = plt.colorbar(CS,ticks=[-60,-30,0,30,60])

    sub = fig.add_subplot(4,4,14)
    CS = sub.contourf(x1coord,x2coord,predY2d-testY2d,50,cmap='seismic')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    cbar = plt.colorbar(CS,ticks=[-0.5,0.0,0.5])

    # Load EBAND (divided by dV to obtain EBAND density)
    with open(saveDir+'EBAND/prediction_sample.pickle', 'rb') as f:
        [testX, testY, predY] = pickle.load(f)

    for i1 in range(ngfft[0]):
        for i2 in range(ngfft[1]):
            predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]/dV
            testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]/dV
    predY2d *= 1.0e3
    testY2d *= 1.0e3

    sub = fig.add_subplot(4,4,7)
    CS = sub.contourf(x1coord,x2coord,testY2d,50) #vmin=0.0035,vmax=0.017
    cbar = plt.colorbar(CS,ticks=[-1.6,-1.2,-0.8,-0.4])
    sub.set_title(r'\begin{center} (c) \\ Band structure energy \\ density ($10^{-3}$ a.u.) \end{center}',fontsize=16,multialignment='center')

    sub = fig.add_subplot(4,4,11)
    CS = sub.contourf(x1coord,x2coord,predY2d,50)
    cbar = plt.colorbar(CS,ticks=[-1.6,-1.2,-0.8,-0.4])

    sub = fig.add_subplot(4,4,15)
    CS = sub.contourf(x1coord,x2coord,predY2d-testY2d,50,cmap='seismic')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    cbar = plt.colorbar(CS,ticks=[-4e-3,-2e-3,0,2e-3])

    # Load ENTR (divided by dV to obtain ENTR density)
    with open(saveDir+'ENTR/prediction_sample.pickle', 'rb') as f:
        [testX, testY, predY] = pickle.load(f)

    for i1 in range(ngfft[0]):
        for i2 in range(ngfft[1]):
            predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]/dV
            testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]/dV
    predY2d *= 1.0e6
    testY2d *= 1.0e6

    sub = fig.add_subplot(4,4,8)
    CS = sub.contourf(x1coord,x2coord,testY2d,50,vmin=-28,vmax=10)
    cbar = clippedcolorbar(CS,ticks=[-20,-10,0,10])
    sub.set_title(r'\begin{center} (d) \\ Volumetric entropy \\ ($10^{-6}$ a.u.) \end{center}',fontsize=16,multialignment='center')

    sub = fig.add_subplot(4,4,12)
    CS = sub.contourf(x1coord,x2coord,predY2d,50,vmin=-28,vmax=10)
    cbar = clippedcolorbar(CS,ticks=[-20,-10,0,10])
    #cbar.ax.set_yticklabels([r'$1\times10^{-8}$',r'$2\times10^{-8}$',r'$3\times10^{-8}$',r'$4\times10^{-8}$'])

    sub = fig.add_subplot(4,4,16)
    CS = sub.contourf(x1coord,x2coord,predY2d-testY2d,50,cmap='seismic')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    cbar = plt.colorbar(CS,ticks=[0,5,10,15])
#    cbar.ax.set_yticklabels(['t','r','y','d'])
    #cbar.ax.set_yticklabels([r'$1\times10^{-8}$',r'$2\times10^{-8}$',r'$3\times10^{-8}$',r'$4\times10^{-8}$'])

    fig.savefig('Figures/sample_prediction.pdf', bbox_inches='tight')

    return

##-------------------------------------------------------------------------------
## Show one example of prediction
##-------------------------------------------------------------------------------
def show_one_prediction(YdataType,saveDir,acell,ngfft):

    # Load one prediction example
    with open(saveDir+YdataType+'/prediction_sample.pickle', 'rb') as f:
        [testX, testY, predY] = pickle.load(f)

    # Compute coordinates
    a = testX[0]
    b = testX[1]
    c = testX[2]
    r = testX[3]
    s = testX[4]
    t = testX[5]
    cos_r = math.cos(math.radians(r))
    cos_s = math.cos(math.radians(s))
    cos_t = math.cos(math.radians(t))
    x1coord = np.empty((ngfft[0],ngfft[1]))
    x2coord = np.empty((ngfft[0],ngfft[1]))
    predY2d = np.empty((ngfft[0],ngfft[1]))
    testY2d = np.empty((ngfft[0],ngfft[1]))

    # Compute volume of each 3D grid
    dV = (acell[0]*acell[1]*acell[2]*a*b*c)/(ngfft[0]*ngfft[1]*ngfft[2]) \
             * math.sqrt( 1 + 2*cos_r*cos_s*cos_t - cos_r*cos_r - cos_s*cos_s - cos_t*cos_t )

    ishift = 0
    if YdataType=='DEN':
        for i1 in range(ngfft[0]):
            for i2 in range(ngfft[1]):
                x1coord[i1,i2] = float(i1)/ngfft[0]*acell[0]*a + float(i2)/ngfft[1]*acell[1]*b*cos_t
                x2coord[i1,i2] = float(i2)/ngfft[1]*acell[1]*b*math.sin(math.radians(t))
                predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]/dV
                testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]/dV
    else:
        for i1 in range(ngfft[0]):
            for i2 in range(ngfft[1]):
                x1coord[i1,i2] = float(i1)/ngfft[0]*acell[0]*a + float(i2)/ngfft[1]*acell[1]*b*cos_t
                x2coord[i1,i2] = float(i2)/ngfft[1]*acell[1]*b*math.sin(math.radians(t))
                predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]
                testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]

    fig = plt.figure(figsize=(16,12))

    sub = fig.add_subplot(231)
    CS = sub.contourf(x1coord,x2coord,predY2d,50) #vmin=0.0035,vmax=0.017
    sub.set_title(YdataType+' (ML)')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS)

    sub = fig.add_subplot(232)
    CS = sub.contourf(x1coord,x2coord,testY2d,50)
    sub.set_title(YdataType+' (true)')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS)

    sub = fig.add_subplot(233)
    CS = sub.contourf(x1coord,x2coord,predY2d-testY2d,50,cmap='seismic')
    sub.set_title(YdataType+' (error)')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS)

    ishift = ngfft[0]*ngfft[1]*int(ngfft[2]/2)
    if YdataType=='DEN':
        for i1 in range(ngfft[0]):
            for i2 in range(ngfft[1]):
                x1coord[i1,i2] = float(i1)/ngfft[0]*acell[0]*a + float(i2)/ngfft[1]*acell[1]*b*cos_t
                x2coord[i1,i2] = float(i2)/ngfft[1]*acell[1]*b*math.sin(math.radians(t))
                predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]/dV
                testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]/dV
    else:
        for i1 in range(ngfft[0]):
            for i2 in range(ngfft[1]):
                x1coord[i1,i2] = float(i1)/ngfft[0]*acell[0]*a + float(i2)/ngfft[1]*acell[1]*b*cos_t
                x2coord[i1,i2] = float(i2)/ngfft[1]*acell[1]*b*math.sin(math.radians(t))
                predY2d[i1,i2] = predY[ ishift+i1+i2*ngfft[0] ]
                testY2d[i1,i2] = testY[ ishift+i1+i2*ngfft[0] ]

    sub = fig.add_subplot(234)
    CS = sub.contourf(x1coord,x2coord,predY2d,50) #vmin=0.0035,vmax=0.017
    sub.set_title(YdataType+' (ML)')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS)

    sub = fig.add_subplot(235)
    CS = sub.contourf(x1coord,x2coord,testY2d,50)
    sub.set_title(YdataType+' (true)')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS)

    sub = fig.add_subplot(236)
    CS = sub.contourf(x1coord,x2coord,predY2d-testY2d,50,cmap='seismic')
    sub.set_title(YdataType+' (error)')
    sub.set_xlabel(r'$x_1$ (Bohr)')
    sub.set_ylabel(r'$x_2$ (Bohr)')
    cbar = plt.colorbar(CS)

    fig.savefig('Figures/sample_'+YdataType+'.pdf', bbox_inches='tight')

    return

##-------------------------------------------------------------------------------
## Plot total ENTR and EBAND
##-------------------------------------------------------------------------------
def plot_total_ENTR_and_EBAND_and_ETOT(firstTestID,testSize,saveDir,volStrain,tsmear):

    # Load true ENTR and EBAND values
    data = np.loadtxt('../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/mg_summarizedOutput.txt', delimiter=',', skiprows=1)
    trueTotalENTR = data[firstTestID-1:firstTestID-1+testSize, 2]/(-tsmear)
    trueTotalEBAND = data[firstTestID-1:firstTestID-1+testSize, 3]

    # EBAND
    YdataType = 'EBAND'
    saveDirY = saveDir + YdataType + '/'
    predY = np.load(saveDirY+'predY.npy')
    predTotalEBAND = np.sum(predY,axis=1)

    # ENTR
    YdataType = 'ENTR'
    saveDirY = saveDir + YdataType + '/'
    predY = np.load(saveDirY+'predY.npy')
    predTotalENTR = np.sum(predY,axis=1)

#    # Figure 1
#    fig = plt.figure(figsize=(10,5))
#
#    sub = fig.add_subplot(121)
#    sub.scatter(volStrain, (predTotalEBAND - trueTotalEBAND)*1000)
#    sub.set_title('EBAND')
#    sub.set_xlabel('Volumetric strain')
#    sub.set_ylabel('Error (mHartree)')
#
#    print('Max eband error (Hartree):', np.max(predTotalEBAND - trueTotalEBAND))
#    print('Mean eband error (Hartree):', np.mean(predTotalEBAND - trueTotalEBAND))
#
#    sub = fig.add_subplot(122)
#    sub.scatter(volStrain, (predTotalENTR - trueTotalENTR)*tsmear*1000)
#    sub.set_title('ENTR')
#    sub.set_xlabel('Volumetric strain')
#    sub.set_ylabel('Error (mHartree)')
#    #sub.set_xlim((0.0,0.1))
#
#    print('Max entr error (Hartree):', np.max(predTotalENTR - trueTotalENTR)*tsmear)
#    print('Mean entr error (Hartree):', np.mean(predTotalENTR - trueTotalENTR)*tsmear)
#
#    fig.savefig('Figures/total_entr_eband_error_vs_vol_strain.pdf', bbox_inches='tight')
#
#    # Figure 2
#    fig = plt.figure(figsize=(10,5))
#
#    sub = fig.add_subplot(121)
#    sub.scatter(volStrain, np.divide(predTotalEBAND-trueTotalEBAND,trueTotalEBAND))
#    sub.set_title('EBAND')
#    sub.set_xlabel('Volumetric strain')
#    sub.set_ylabel('Relative error')
#
#    sub = fig.add_subplot(122)
#    sub.scatter(volStrain, np.divide(predTotalENTR-trueTotalENTR,trueTotalENTR))
#    sub.set_title('ENTR')
#    sub.set_xlabel('Volumetric strain')
#    sub.set_ylabel('Relative error')
#
#    fig.savefig('Figures/total_entr_eband_rel_error_vs_vol_strain.pdf', bbox_inches='tight')

    # Figure 3
    fig = plt.figure(figsize=(10,3))
    plt.subplots_adjust(wspace=0.8, hspace=0.5)

    error = (np.abs(predTotalEBAND - trueTotalEBAND))*1000.0
    print('Max error (total band energy) is ',np.max(error),'mHartree')
    print('Mean error (total band energy) is ',np.mean(error),'mHartree')
    sub = fig.add_subplot(131)
    sub.scatter(trueTotalEBAND, predTotalEBAND, alpha=0.5, clip_on=False)
#    lims = [
#        np.min([sub.get_xlim(), sub.get_ylim()]),  # min of both axes
#        np.max([sub.get_xlim(), sub.get_ylim()]),  # max of both axes
#        ]
    lims = [-0.85,-0.65]
    sub.set_xlim(lims)
    sub.set_ylim(lims)
    sub.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    sub.set_title(r'\begin{center}(a) \\ Band structure energy \\ (Ha) \end{center}',fontsize=16)
    sub.set_xlabel('Ground truth')
    sub.set_ylabel('ML prediction')
    sub.set_xticks([-0.85,-0.75,-0.65])
    sub.set_yticks([-0.85,-0.75,-0.65])
    sub.set_aspect('equal')

    error = (np.abs(predTotalENTR - trueTotalENTR))*tsmear*1000.0
    print('Max error (total entropic energy) is ',np.max(error),'mHartree')
    print('Mean error (total entropic energy) is ',np.mean(error),'mHartree')
    sub = fig.add_subplot(132)
    sub.scatter(-trueTotalENTR*tsmear*1000, -predTotalENTR*tsmear*1000, alpha=0.5, clip_on=False)
    lims = [-2e-1,2e-1]
    sub.set_xlim(lims)
    sub.set_ylim(lims)
    sub.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    sub.set_title(r'\begin{center} (b) \\ Entropic energy \\ (mHa) \end{center}',fontsize=16)
    sub.set_xlabel('Ground truth')
    sub.set_ylabel('ML prediction')
    sub.set_xticks([-2e-1,0,2e-1])
    sub.set_yticks([-2e-1,0,2e-1])
    sub.set_aspect('equal')

    mlEnergy = np.load('../06_ML_ETOT/Saved_ml_results_occopt4_train2000/ETOT/predY.npy')
    mlEnergy = mlEnergy.reshape((testSize))
    abinitEnergy = np.loadtxt('../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/mg_summarizedOutput.txt', delimiter=',', skiprows=1)
    abinitEnergy = abinitEnergy[firstTestID-1:firstTestID-1+testSize, 1]
    error = (np.abs(mlEnergy - abinitEnergy))*1000.0
    print('Max error (total free energy) is ',np.max(error),'mHartree')
    print('Mean error (total free energy) is ',np.mean(error),'mHartree')
    sub = fig.add_subplot(133)
    sub.scatter(abinitEnergy, mlEnergy, alpha=0.5)
    lims = [-3.61,-3.57]
    sub.set_xlim(lims)
    sub.set_ylim(lims)
#    lims = [
#        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#        ]
    sub.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    sub.set_aspect('equal')
    sub.set_xlabel('Ground truth')
    sub.set_ylabel('ML prediction')
    sub.set_title(r'\begin{center} (c) \\ Total free energy \\ (Ha) \end{center}',fontsize=16)
    sub.set_xticks([-3.61,-3.59,-3.57])
    sub.set_yticks([-3.61,-3.59,-3.57])
    sub.set_aspect('equal')

    fig.savefig('Figures/ml_vs_true_eband_entr_etot.pdf', bbox_inches='tight')

    return

##-------------------------------------------------------------------------------
## Plot displacement error for XRED
##-------------------------------------------------------------------------------
def plot_disp_error_for_XRED(testSize,saveDirXRED,volStrain):

    # Displacement error
    dispError = np.loadtxt(saveDirXRED+'XRED/ml_test_disp_error.txt')
    fig,ax = plt.subplots(1,2,figsize=(8,2.5))
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    ax[0].scatter(volStrain,dispError,alpha=0.5,clip_on=False)
    ax[0].set_xlabel('Volumetric strain')
    ax[0].set_ylabel(r'\begin{center} Absolute displacement \\ error (Bohr) \end{center}')
    ax[0].set_ylim((0.0,0.3))
    ax[0].set_title('(a)',fontsize=16)
    ax[0].set_xlim((-0.3,0.3))
    ax[0].xaxis.set_major_locator(MultipleLocator(0.3))
    ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))


#    # True displacement
#    trueDisp = np.loadtxt(saveDirXRED+'XRED/ml_test_true_disp.txt')
#    sub = fig.add_subplot(122)
#    sub.scatter(volStrain,trueDisp)
#    sub.set_title('XRED')
#    sub.set_xlabel('Volumetric strain')
#    sub.set_ylabel('True displacement (Bohr)')

    # Average displacement error vs training size
    YdataType = 'XRED'
    saveDirXRED = saveDirXRED.split('/')
    saveDirXRED = saveDirXRED[0] + '/' + saveDirXRED[1] + '/'
    trainSize = np.linspace(0,2000,9)
    trainSize = trainSize[1:]
    meanError = []
    for i in range(np.shape(trainSize)[0]):
        saveDirY = saveDirXRED + 'Saved_ml_results_occopt4_train' + str(int(trainSize[i])) + '/' + YdataType + '/'
        error = np.loadtxt(saveDirY+'ml_test_disp_error.txt')
        meanError.append(np.mean(error))

    ax[1].plot(trainSize, meanError, 'kx--',clip_on=False)
    ax[1].set_xlim((0,2000))
    ax[1].set_ylim((0.0,0.010))
    ax[1].set_title('(b)',fontsize=16)
    ax[1].set_xlabel('Training size')
    ax[1].set_ylabel(r'\begin{center} Mean displacement \\ error (Bohr) \end{center}')

    fig.savefig('Figures/ml_xred_disp_error.pdf', bbox_inches='tight')

    return

##-------------------------------------------------------------------------------
## Clipped colorbar
##-------------------------------------------------------------------------------
def clippedcolorbar(CS, **kwargs):
    from matplotlib.cm import ScalarMappable
    from numpy import arange, floor, ceil
    fig = CS.ax.get_figure()
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin<vmin
    clipupper = CS.zmax>vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend']=='neither'
    # set the colorbar boundaries
    boundaries = arange((floor(vmin/step)-1+1*(cliplower and noextend))*step, (ceil(vmax/step)+1-1*(clipupper and noextend))*step, step)
    kwargs['boundaries'] = boundaries
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min','max']:
        extend_min = cliplower or ( 'extend' in kwargs.keys() and kwargs['extend']=='min' )
        extend_max = clipupper or ( 'extend' in kwargs.keys() and kwargs['extend']=='max' )
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    return fig.colorbar(m, **kwargs)

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Variables from Abinit
    ngfft = [36,64,60]
    nfft = ngfft[0]*ngfft[1]*ngfft[2]
    acell = [5.89,10.201779256580686,9.56]
    nband = 20
    tsmear = 0.01

    # Directory where all ml results are saved
    saveDir = '../01_ML_fields/Saved_ml_results_occopt4_train2000/'
    saveDirXRED = '../02_ML_XRED/Saved_ml_results_occopt4_train2000/'
#
#    # Create directory for figures
#    create_directory('Figures/')

    # Compute volumetric strain
    testX = np.load(saveDir+'testX.npy')
    volStrain = compute_vol_strain_for_testX(testX)
    
    # Plot relative error vs volumetric strain
    plot_error_vs_vol_strain(saveDir,volStrain)
    plot_error_vs_vol_strain2(saveDir,volStrain)

    # Plot relative error vs train size
    plot_error_vs_train_size(saveDir)

    # Plot one prediction
    show_one_prediction_for_all_variables(saveDir,acell,ngfft) 
#    show_one_prediction('DEN',saveDir,acell,ngfft)
#    show_one_prediction('VCLMB',saveDir,acell,ngfft)
#    show_one_prediction('ENTR',saveDir,acell,ngfft)
#    show_one_prediction('EBAND',saveDir,acell,ngfft)

#    # Check sum of ENTR and EBAND and compare with Abinit results
    firstTestID = 2001
    testSize = np.shape(testX)[0]
    plot_total_ENTR_and_EBAND_and_ETOT(firstTestID,testSize,saveDir,volStrain,tsmear)    

    # Plot displacement error
    plot_disp_error_for_XRED(testSize,saveDirXRED,volStrain)

if __name__ == '__main__': main()
