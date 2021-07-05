#!/usr/bin/env python3

#################################################################################
# Extract and save stress values from Abinit output files
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

#---------------------------------------------------------------------------
# Extract stress value from each Abinit output file
#---------------------------------------------------------------------------
def extract_stress_value(filename,stressDirection):

    checkLine = 'sigma(%i %i)' %(stressDirection,stressDirection)
    with open(filename) as inputFile:
        flag = False
        flag2 = False
        for line in inputFile:
            if 'gradients are converged' in line:
                flag = True
            elif flag and 'Cartesian components of stress tensor (hartree/bohr^3)' in line:
                flag2 = True
            elif flag2 and checkLine in line:
                stress = line.split()[2]
                break

    return stress

#---------------------------------------------------------------------------
# Collate all stress values
#---------------------------------------------------------------------------
def collate_stress_values_for_each_direction(stressDirection):

    f = open('Abinit_runs/direction'+str(stressDirection)+'/mg_summarizedStressValues.txt','w')
    for idata in range(21):
        filename = 'Abinit_runs/direction'+str(stressDirection)+'/mg'+str(idata+1)+'.out'
        stress = extract_stress_value(filename,stressDirection)
        f.write(stress+'\n')
    f.close()

    return

#---------------------------------------------------------------------------
# Extract strain value from each Abinit output file
#---------------------------------------------------------------------------
def extract_strain_value(filename,stressDirection):

    with open(filename) as inputFile:
        flag = False
        for line in inputFile:
            if 'END DATASET(S)' in line:
                flag = True
            elif flag and 'acell' in line:
                deformedStrain = line
                break
        deformedStrain = deformedStrain.strip('            acell')
        deformedStrain = deformedStrain.strip(' Bohr\n')

    return deformedStrain

#---------------------------------------------------------------------------
# Collate all strain values
#---------------------------------------------------------------------------
def collate_strain_values_for_each_direction(stressDirection):

    f = open('Abinit_runs/direction'+str(stressDirection)+'/mg_summarizedStrainValues.txt','w')
    for idata in range(21):
        filename = 'Abinit_runs/direction'+str(stressDirection)+'/mg'+str(idata+1)+'.out'
        deformedStrain = extract_strain_value(filename,stressDirection)
        f.write(deformedStrain+'\n')
    f.close()
    data = np.loadtxt('Abinit_runs/direction'+str(stressDirection)+'/mg_summarizedStrainValues.txt')
    data[:,0] /= acell[0]
    data[:,1] /= acell[1]
    data[:,2] /= acell[2]
    data = data - 1.0
    np.savetxt('Abinit_runs/direction'+str(stressDirection)+'/mg_summarizedStrainValues.txt',data)

    return

#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
acell = [5.89,10.201779256580686,9.56]

def main():

    collate_stress_values_for_each_direction(1)
    collate_stress_values_for_each_direction(2)
    collate_stress_values_for_each_direction(3)

    collate_strain_values_for_each_direction(1)
    collate_strain_values_for_each_direction(2)
    collate_strain_values_for_each_direction(3)

if __name__ == '__main__': main()
