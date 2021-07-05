#!/usr/bin/env python3

#################################################################################
## Generate files to run Abinit
## Create by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
import os, shutil
import math

#---------------------------------------------------------------------------
# Make directory
#---------------------------------------------------------------------------
def makeDirectory(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    return

#---------------------------------------------------------------------------
# Get xred
#---------------------------------------------------------------------------
def getXred(xyz):
    xredLines = """%.6f %.6f %.6f
  %.6f %.6f %.6f
  %.6f %.6f %.6f
  %.6f %.6f %.6f
"""%(xyz[0], xyz[1], xyz[2], \
     0.5+xyz[0], 0.5+xyz[1], xyz[2], \
     -xyz[0], 2.0/3.0-xyz[1], 0.5-xyz[2], \
     0.5-xyz[0], 1.0/6.0-xyz[1], 0.5-xyz[2])

    return xredLines

#--------------------------------------------------------------------------------
# Generate Abinit input file
# Note: there are two sets of Abinit run
# Run 1: use nonscf (but nonscf does not compute energy)
# Run 2: use DEN from previous run to compute energy with zero scf.
#--------------------------------------------------------------------------------
def saveAbinitInputFile(abcrst, xredLines, filename):

    # Mg lattice parameters
    #La = 6.0419031073
    #Lb = 10.464702002
    #Lc = 9.8453372801
    La = 5.89
    Lb = 10.201779256580686
    Lc = 9.56

    fileContent = """
  # Magnesium: Structural Optimization

  ndtset 3
  # Output parameters 
  prtwf1 1
  prtwf2 0
  prtwf3 0
  prtden 0
  prteig 0

  # Occupation options 
  occopt 4
  tsmear 0.01
  nband 20

  #Definition of the unit cell
  acell %.12f %.12f %.12f
  angdeg %.4f %.4f %.4f
  chkprim 0

  #Definition of the atom types
  ntypat 1          
  znucl 12          

  #Definition of the atoms
  natom 4           
  typat 1 1 1 1
  xred
  %s

  #Definition of the planewave basis set
  ecut  24.0         # Maximal kinetic energy cut-off, in Hartree

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
  iscf1 -2
  nstep1 50
  nstep2 0          # Maximal number of SCF cycles
  nstep3 0
  tolwfr1 1.0d-12
  toldfe2 1.0d-10
  toldfe3 1.0d-10
  getwfk1 0
  getwfk2 -1
  getwfk3 -2
  getden1 -1
  getden2 -1
  getden3 0
""" %(La*abcrst[0],Lb*abcrst[1],Lc*abcrst[2],abcrst[3],abcrst[4],abcrst[5],xredLines)

    f = open(filename,'w')
    f.write(fileContent)
    f.close()

    return

#--------------------------------------------------------------------------------
# Generate bash script
#--------------------------------------------------------------------------------
def generateScript(testSize):
    content ="""#!/bin/sh

cd tmp

for NUM in $(seq 1 %i); do

NAME=pred${NUM}
echo $NAME

cp ../Electron_density_binary_files/${NAME}o_DEN ../Electron_density_binary_files/${NAME}i_DS1_DEN
cp ../Electron_density_binary_files/${NAME}o_DEN ../Electron_density_binary_files/${NAME}o_DS1_DEN

cat > abinit.files << EOF
../AbinitInput/${NAME}.in
../AbinitOutput/${NAME}.out
../Electron_density_binary_files/${NAME}i
../Electron_density_binary_files/${NAME}o
tmp
../../00_AbinitRun/Psps/mg.pseu
EOF

mpirun -np 18 abinit < abinit.files > log_abinit
done

""" %(testSize)

    f = open('4_runAbinit.sh','w')
    f.write(content)
    f.close()

    return


#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------
def main():
    testX = np.load('../01_ML_fields/Saved_ml_results_occopt4_train2000/testX.npy')
    testSize = np.shape(testX)[0]

    predXred = np.load('../02_ML_XRED/Saved_ml_results_occopt4_train2000/XRED/predY.npy')

    makeDirectory('AbinitInput/')
    makeDirectory('AbinitOutput/')
    makeDirectory('tmp/')

    for index in range(testSize):
        abcrst = testX[index,:]
        xyz = predXred[index,:]
        xredLines = getXred(xyz)
        abinitInputFilename = 'AbinitInput/pred' + str(index+1) + '.in'
        saveAbinitInputFile(abcrst,xredLines,abinitInputFilename)

    generateScript(testSize)

if __name__ == '__main__': main()
