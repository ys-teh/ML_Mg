#/usr/bin/env python3

#################################################################################
# Run Abinit to generate stress-strain curve for comparison with ML-generated curve
# Create by Ying Shi Teh
# Last modified on July 5, 2021
#################################################################################

import numpy as np
import math
import os

#--------------------------------------------------------------------------------
# Generate Abinit input file
#--------------------------------------------------------------------------------
def saveAbinitInputFile(axisX,acellLine,filename):

    fileContent = """
  # Magnesium: Structural Optimization

  # Output parameters 
  prtwf 1
  prtden 1
  prteig 0
  prtvclmb 1
  prtvha 1
  iomode 3

  # Optimization parameters
  optcell %i
  ionmov 2
  ntime 50
  dilatmx 1.15
  ecutsm 0.5 # might need to change it
  tolmxf 5.0d-6
  chksymbreak 0

  # Occupation options 
  occopt 4
  tsmear 0.01
  nband 20

  #Definition of the unit cell
  acell %s
  angdeg 90.0 90.0 90.0
  chkprim 0

  #Definition of the atom types
  ntypat 1          
  znucl 12          

  #Definition of the atoms
  natom 4           
  typat 1 1 1 1
  xred
  0.0 0.0 0.0
  0.5 0.5 0.0
  0.0 2/3 0.5
  0.5 1/6 0.5

  #Definition of the planewave basis set
  ecut 24.0         # Maximal kinetic energy cut-off, in Hartree

  #Exchange-correlation functional
  ixc 7            # Perdew_Wang LDA

  #Definition of the k-point grid
  kptopt 1
  ngkpt 12 12 12
  nshiftk 1
  shiftk 0.0 0.0 0.0

  #Definition of the number of grid points for FFT
  ngfft 36 64 60

  #Definition of the SCF procedure
  nstep 100          # Maximal number of SCF cycles
  toldff 5.0d-7
""" %(axisX+7,acellLine)

    f = open(filename,'w')
    f.write(fileContent)
    f.close()

    return

#---------------------------------------------------------------------------
# Generate SBATCH scripts
#---------------------------------------------------------------------------
def generateScript(dirName):
    lines = """#!/bin/sh

############################################################################
# RUN ABINIT
############################################################################

mpirun -np 18 abinit < %s/mg.files > log_abinit

echo Done!
"""%(dirName)
    
    filename = dirName + 'run.sh'
    f = open(filename,'w')
    f.write(lines)
    f.close()

    return

#---------------------------------------------------------------------------
# Extract acell
#---------------------------------------------------------------------------
def extract_acell(filename):
    with open(filename) as inputFile:
        flag = False
        for line in inputFile:
            if 'END DATASET' in line:
                flag = True
            elif flag:
                if 'acell' in line:
                    acell = line.split()[1:4]
                    break

    return acell

#---------------------------------------------------------------------------
# Save mg files
#---------------------------------------------------------------------------
def saveMgFiles(i,dirName):
    lines = """%smg%i.in
%smg%i.out
%smg%ii
%smg%io
%smg%i
../00_AbinitRun/Psps/mg.pseu

"""%(dirName,i,dirName,i,dirName,i,dirName,i,dirName,i)

    filename = dirName + 'mg.files'
    f = open(filename,'w')
    f.write(lines)
    f.close()

    return

#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------
def main():

    # Directory name
    dirName = 'Abinit_runs/'

    # Mg lattice parameters
    L = [5.89,10.201779256580686,9.56]

    # Generate run.sh script
    generateScript(dirName)

    # Generate X and save file as mg_X.npy
    N = 21
    axisX = 0
    varyX = np.linspace(0.9,1.1,N)
    X = np.kron(np.array([1.0,1.0,1.0,90,90,90]),np.ones((N,1)))
    X[:,axisX] = varyX

    # Generate Abinit input file and run
    for i in range(9,0,-1):
        print('Run ',i)
        abcrst = X[i-1,:]

        # Extract acell from previous run
        if i<11:
            filename = dirName + 'mg%i.out' %(i+1)
        else:
            filename = dirName + 'mg%i.out' %(i-1)

        # Change acell on current run
        acell = extract_acell(filename)
        acell[axisX] = str(L[axisX]*varyX[i-1])
        acellLine = acell[0] + ' ' + acell[1] + ' ' + acell[2]
        print(acellLine)
        filename = dirName + 'mg%i.in' %(i)
        saveAbinitInputFile(axisX,acellLine,filename)
        saveMgFiles(i,dirName)

        # Run Abinit file
        runLine = 'mpirun -np 18 abinit < Abinit_runs/mg.files > log_abinit'
        os.system(runLine)

if __name__ == '__main__': main()
