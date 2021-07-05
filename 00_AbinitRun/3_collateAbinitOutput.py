#/usr/bin/env python3

#################################################################################
## Post-processing for Mg
## Collate useful information from Abinit .out files
## Copy useful files to folder Output/
## Delete other files on your own later
## Created by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import os
import math
from shutil import copyfile
import pandas as pd
import numpy as np

#--------------------------------------------------------------------------------
# Get iteration number, etotal
#--------------------------------------------------------------------------------
def getOutputInfo(filename):
    # Get iteration number, etotal
    with open(filename) as inputFile:
        flag = False
        for line in inputFile:
            if 'gradients are converged' in line:
                iterNumber = line
                iterNumber = iterNumber.lstrip('At Broyd/MD step  ')
                iterNumber = iterNumber.rstrip(', gradients are converged :\n')
                continue
            if 'entropy' in line:
                entr = line
                entr = line.split()[2]
                continue
            if 'Band energy' in line:
                eband = line
                eband = line.split()[7]
                continue
            if 'after computation' in line:
                flag = True
                continue
            if 'etotal' in line and flag:
                etotal = line.split()[1]
                break

    return iterNumber, etotal, entr, eband

#--------------------------------------------------------------------------------
# Copy output files to folder
#--------------------------------------------------------------------------------
def copyOutputFiles(directory, prefix, iterNumber):
    filename = directory + prefix + 'o_TIM' + iterNumber + '_DEN.nc'
    saveFilename = 'Abinit_output/' + prefix + '_DEN.nc'
    copyfile(filename, saveFilename)

    filename = directory + prefix + 'o_TIM' + iterNumber + '_VCLMB.nc'
    saveFilename = 'Abinit_output/' + prefix + '_VCLMB.nc'
    copyfile(filename, saveFilename)

    filename = directory + prefix + 'o_TIM' + iterNumber + '_VHA.nc'
    saveFilename = 'Abinit_output/' + prefix + '_VHA.nc'
    copyfile(filename, saveFilename)

    filename = directory + prefix + '.out'
    saveFilename = 'Abinit_output/' + prefix + '.out'
    copyfile(filename, saveFilename)

    filename = directory + prefix + 'o_WFK.nc'
    saveFilename = 'Abinit_output/' + prefix + 'o_WFK.nc'
    copyfile(filename, saveFilename)

    return

#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------
def main():
    batchSize = 100
    dataSize = 3000
    startDataNumber = 1
    totalBatches = int(math.ceil(float(dataSize)/float(batchSize)))

    # Make directory
    path = 'Abinit_output/'
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    # Loop through
    f = open('mg_summarizedOutput.txt','w')
    f.write('number of iterations, total energy, entropy, band energy\n')
    for number in range(1,dataSize+1):
        batchNumber = int(math.ceil(float(number)/float(batchSize)))
        directory = 'batch' + str(batchNumber) + '/Output/'
        prefix = 'mg' + str(number+startDataNumber-1)
        filename = directory + prefix + '.out'
        print(filename)
        iterNumber, etotal, entr, eband = getOutputInfo(filename)
        copyOutputFiles(directory, prefix, iterNumber)
        f.write(iterNumber + ', ' + etotal + ', ' + entr + ', ' + eband + '\n');
    f.close()

if __name__ == '__main__': main()
