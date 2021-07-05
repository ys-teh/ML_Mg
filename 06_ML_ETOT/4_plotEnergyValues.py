#!/usr/bin/env python3

#################################################################################
# Plot energy values
# Create by Ying Shi Teh
# Last modified on July 5, 2021
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

#---------------------------------------------------------------------------
# Plot energy errors
#---------------------------------------------------------------------------
def compute_energy_error(firstTestID,testSize):
    mlEnergy = np.load('Saved_ml_results_occopt4_train2000/ETOT/predY.npy')
    mlEnergy = mlEnergy.reshape((testSize))
    abinitEnergy = np.loadtxt('../00_AbinitRun/saved_data_strain10p_gcutoff2_occopt4/mg_summarizedOutput.txt', delimiter=',', skiprows=1)
    abinitEnergy = abinitEnergy[firstTestID-1:firstTestID-1+testSize, 1]
    error = (np.abs(mlEnergy - abinitEnergy))*1000.0
    print('Max error is ',np.max(error),'mHartree')
    print('Mean error is ',np.mean(error),'mHartree')

    fig = plt.figure(figsize=(4,3))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    sub = fig.add_subplot(111)
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
    sub.set_xlabel('True total energy (Hartree)')
    sub.set_ylabel('ML total energy (Hartree)')
    sub.set_title('(a)')
    sub.set_xticks([-3.61,-3.59,-3.57])
    sub.set_yticks([-3.61,-3.59,-3.57])

    plt.savefig('Figures/ml_vs_true_energy3.pdf', bbox_inches='tight')

    #plt.show()

    return

#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
def main():

    # Some parameters    
    testSize = 1000
    firstTestID = 2001

    # Create directory for figures
    create_directory('Figures/')

    # Collate energy values and compare with the 
    compute_energy_error(firstTestID,testSize)
    
if __name__ == '__main__': main()
