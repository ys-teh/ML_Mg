#!/usr/bin/env python3

#################################################################################
## Collate energy values (predicted and actual) for Mg
## Create by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
import os, shutil
import math
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})
from matplotlib import pyplot as plt
import netCDF4

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
# Compute volumetric strain for testX
#--------------------------------------------------------------------------------
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

    return volStrain

#---------------------------------------------------------------------------
# Extract energy and force values from Abinit output
# First column shows Abinit energy
# Second column shows predicted energy
#---------------------------------------------------------------------------
def collate_ml_values(testSize):
    f1 = open('AbinitOutput/summarizedEnergyValues.txt','w')
    f2 = open('AbinitOutput/summarizedForceValues.txt','w')

    for index in range(testSize):
        filename = 'AbinitOutput/pred'+str(index+1)+'.out'
    
        with open(filename) as input_file:
            for line in input_file:
                if 'etotal2' in line:
                    energy1 = line
                elif 'etotal3' in line:
                    energy2 = line
                elif 'fcart2' in line:
                    force1 = line
                elif 'fcart3' in line:
                    force2 = line                 
                    break
        energy1 = energy1.split()[1]
        energy2 = energy2.split()[1]
        line = energy1 + ' ' + energy2 + '\n'
        f1.write(line)

        force1 = force1.split()
        force2 = force2.split()
        line = force1[1] + ' ' + force1[2] + ' ' + force1[3] + ' ' \
               + force2[1] + ' ' + force2[2] + ' ' + force2[3] + '\n'
        f2.write(line)

    f1.close()
    f2.close()

    return

#---------------------------------------------------------------------------
# Plot forces
#---------------------------------------------------------------------------
def plot_force(volStrain):
    mlForce = np.loadtxt('AbinitOutput/summarizedForceValues.txt')
    maxForce = np.amax(np.abs(mlForce[:,:3]),axis=1)

    fig,ax = plt.subplots(figsize=(3.4,3))
    ax.scatter(volStrain,maxForce*1000.0,alpha=0.5)
    ax.set_xlabel('Volumetric strain')
    ax.set_ylabel('Maximal absolute force \n ($10^{-3}$ Hartree/Bohr)')
    ax.set_ylim([0.0,2.0])
#    ax.text(0.8, 1.0, 'Average = %f Hartree/Bohr' %np.mean(maxForce))
    fig.savefig('Figures/force_vs_vol_strain.pdf', bbox_inches='tight')

    #plt.show()

    return

#---------------------------------------------------------------------------
# Extract energy values from Abinit output2
# First column shows Hartree energy
# Second column shows XC energy
# Third column shows Ewald energy
# Fourth column shows PspCore energy
#---------------------------------------------------------------------------
def collate_energy_values_from_ml_den(testSize):
    f1 = open('AbinitOutput2/summarizedEnergyValues.txt','w')

    for index in range(testSize):
        filename = 'AbinitOutput2/pred'+str(index+1)+'.out'

        flag = False
        with open(filename) as input_file:
            for line in input_file:
                if '== DATASET  2 ==' in line:
                    flag = True
                if flag:
                    if 'Hartree energy' in line:
                        energy1 = line
                        continue
                    elif 'XC energy' in line:
                        energy2 = line
                        continue
                    elif 'Ewald energy' in line:
                        energy3 = line
                        continue
                    elif 'PspCore energy' in line:
                        energy4 = line
                        break
        energy1 = energy1.split()[3]
        energy2 = energy2.split()[3]
        energy3 = energy3.split()[3]
        energy4 = energy4.split()[3]
        line = energy1 + ' ' + energy2 + ' ' + energy3 + ' ' + energy4 + '\n'
        f1.write(line)

    f1.close()

    return

#---------------------------------------------------------------------------
# Compute total energy solely based on ML results
#---------------------------------------------------------------------------
def compute_total_energy_from_ml_results(firstTestID,testSize,saveDir,tsmear):

    # EBAND
    YdataType = 'EBAND'
    saveDirY = saveDir + YdataType + '/'
    predY = np.load(saveDirY+'predY.npy')
    predTotalEBAND = np.sum(predY,axis=1)
    predTotalEBAND = predTotalEBAND[:testSize]

    # ENTR
    YdataType = 'ENTR'
    saveDirY = saveDir + YdataType + '/'
    predY = np.load(saveDirY+'predY.npy')
    predTotalENTR = np.sum(predY,axis=1)
    predTotalENTR = predTotalENTR[:testSize]

    # DEN
    YdataType = 'DEN'
    saveDirY = saveDir + YdataType + '/'
    predDEN = np.load(saveDirY+'predY.npy')

    # Compute \int Vxc(r)*rho(r)dr for each case
    predVxcDEN = np.zeros((testSize))
    for i in range(testSize):
        filename = 'AbinitOutput2/pred' + str(i+1) + '_VXC.nc'
        with netCDF4.Dataset(filename,'r') as nc:
            Vxc = nc.variables['exchange_correlation_potential'][:]
            Vxc = Vxc.reshape((np.size(Vxc)))
        predVxcDEN[i] = np.sum(np.multiply(predDEN[i,:],Vxc))

    # Load the following saved energy values:
    # Hartree energy, XC energy, Ewald energy, PspCore energy
    predEnergies = np.loadtxt('AbinitOutput2/summarizedEnergyValues.txt')
    predEnergies = predEnergies[:testSize]

    # Total energy 
    predIntEnergy = predTotalEBAND - predVxcDEN - predEnergies[:,0] + np.sum(predEnergies[:,1:4],axis=1)
    predTotalEnergy = predIntEnergy + predTotalENTR*(-tsmear)

    return predTotalEnergy

#---------------------------------------------------------------------------
# Plot energy errors
#---------------------------------------------------------------------------
def compute_energy_error(predTotalEnergy,firstTestID,testSize,volStrain):

    # Compare with abinit energy
    abinitEnergy = np.loadtxt('../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/mg_summarizedOutput.txt', delimiter=',', skiprows=1)
    abinitEnergy = abinitEnergy[firstTestID-1:firstTestID-1+testSize, 1]

    # Direct approach
    mlEnergyDirect = np.load('../06_ML_ETOT/Saved_ml_results_occopt4_train2000/ETOT/predY.npy')
    mlEnergyDirect = mlEnergyDirect.reshape((testSize))
    error1 = (np.abs(mlEnergyDirect - abinitEnergy))*1000.0
    print('Max error1 is ',np.max(error1),'mHartree')
    print('Mean error1 is ',np.mean(error1),'mHartree')

    # Sum approach
    error2 = (np.abs(predTotalEnergy - abinitEnergy))*1000.0
    print('Max error2 is ',np.max(error2),'mHartree')
    print('Mean error2 is ',np.mean(error2),'mHartree')

    # Orbital and SCF approaches
    mlEnergy = np.loadtxt('AbinitOutput/summarizedEnergyValues.txt')
    abinitEnergy = np.loadtxt('../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/mg_summarizedOutput.txt', delimiter=',', skiprows=1)
    abinitEnergy = abinitEnergy[firstTestID-1:firstTestID-1+testSize, 1]
    error3 = (np.abs(mlEnergy[:,0] - abinitEnergy))*1000.0
    error4 = (np.abs(mlEnergy[:,1] - abinitEnergy))*1000.0
    print('Max error3 is ',np.max(error3),'mHartree')
    print('Mean error3 is ',np.mean(error3),'mHartree')
    print('Max error4 is ',np.max(error4),'mHartree')
    print('Mean error4 is ',np.mean(error4),'mHartree')

    # Plot results
    fig = plt.figure(figsize=(12,3.3))
    plt.subplots_adjust(wspace=0.6, hspace=0.5)

    sub = fig.add_subplot(141)
    sub.scatter(abinitEnergy, mlEnergyDirect, alpha=0.5)
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
    sub.set_title(r'\begin{center} (a) \\ Direct approach \end{center}',fontsize=16)
    sub.set_xticks([-3.61,-3.59,-3.57])
    sub.set_yticks([-3.61,-3.59,-3.57])

    sub = fig.add_subplot(142)
    sub.scatter(abinitEnergy, predTotalEnergy, alpha=0.5)
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
    sub.set_title(r'\begin{center} (b) \\ Sum approach \end{center}',fontsize=16)
    sub.set_xticks([-3.61,-3.59,-3.57])
    sub.set_yticks([-3.61,-3.59,-3.57])

    sub = fig.add_subplot(143)
    sub.scatter(abinitEnergy, mlEnergy[:,0], alpha=0.5)
    lims = [-3.61,-3.57]
    sub.set_xlim(lims)
    sub.set_ylim(lims)
    sub.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    sub.set_aspect('equal')
    sub.set_xlabel('Ground truth')
    sub.set_title(r'\begin{center} (c) \\ Orbital approach \end{center}',fontsize=16)
    sub.set_xticks([-3.61,-3.59,-3.57])
    sub.set_yticks([-3.61,-3.59,-3.57])

    sub = fig.add_subplot(144)
    sub.scatter(abinitEnergy, mlEnergy[:,1], alpha=0.5)
    lims = [-3.61,-3.57]
    sub.set_xlim(lims)
    sub.set_ylim(lims)
    sub.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    sub.set_aspect('equal')
    sub.set_xlabel('Ground truth')
    sub.set_title(r'\begin{center} (d) \\ SCF approach \end{center}',fontsize=16)
    sub.set_xticks([-3.61,-3.59,-3.57])
    sub.set_yticks([-3.61,-3.59,-3.57])

    plt.suptitle('Total free energy in Ha',fontsize=18)
    plt.savefig('Figures/ml_vs_true_energy.pdf', bbox_inches='tight')

    #plt.show()

    return

#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
def main():

    # Some parameters    
    testSize = 1000
    firstTestID = 2001
    saveDir = '../01_ML_fields/Saved_ml_results_occopt4_train2000/'
    tsmear = 0.01

    # Create directory for figures
    create_directory('Figures/')

    # Compute volumetric strain
    testX = np.load(saveDir+'testX.npy')
    volStrain = compute_vol_strain_for_testX(testX)

    # Collate energy values 
    collate_ml_values(testSize)
    plot_force(volStrain)

    # Collate and compute energy values 
    collate_energy_values_from_ml_den(testSize)
    predTotalEnergy = compute_total_energy_from_ml_results(firstTestID,testSize,saveDir,tsmear)

    # Compare energy values
    compute_energy_error(predTotalEnergy,firstTestID,testSize,volStrain)
 
if __name__ == '__main__': main()
