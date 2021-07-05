#/usr/bin/env python3

#################################################################################
## Generate X data, i.e. a,b,c,r,s,t, following normal distribution
## At zero deformation, a = b = c = 1, r = s = t = 90deg
## Created by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
import math
import os

#--------------------------------------------------------------------------------
# Generate dataSize number of data points on normal distribution with proper cutoff
#--------------------------------------------------------------------------------
def generateOneX(dataSize, cutoff):
    nTemp = math.ceil(dataSize*1.2)
    x = np.random.randn(nTemp)
    deleteIDs = [];
    for i in range(nTemp):
        if x[i] < -cutoff or x[i] > cutoff:
            deleteIDs.append(i)
 
    # Check that after deletions, there are enough data left
    if len(deleteIDs) > math.ceil(0.2*dataSize):
        print('Not enough data!')
    x = np.delete(x,deleteIDs)
    x = x[:dataSize]

    return x

#--------------------------------------------------------------------------------
# Generate dataSize number of data points on normal distribution with proper cutoff
#--------------------------------------------------------------------------------
def generateAllX(dataSize, cutoff):

    # X
    X = np.zeros((dataSize,6));

    # a
    x_min = 0.9
    x_max = 1.1
    x = generateOneX(dataSize, cutoff)
    X[:,0] = 1.0 + x*((x_max-x_min)/cutoff/2.0)

    # b
    x_min = 0.9
    x_max = 1.1
    x = generateOneX(dataSize, cutoff)
    X[:,1] = 1.0 + x*((x_max-x_min)/cutoff/2.0)

    # c
    x_min = 0.9
    x_max = 1.1
    x = generateOneX(dataSize, cutoff)
    X[:,2] = 1.0 + x*((x_max-x_min)/cutoff/2.0)

    # r
    x_min = 84
    x_max = 96
    x = generateOneX(dataSize, cutoff)
    X[:,3] = 90 + x*((x_max-x_min)/cutoff/2.0)

    # s
    x_min = 84
    x_max = 96
    x = generateOneX(dataSize, cutoff)
    X[:,4] = 90 + x*((x_max-x_min)/cutoff/2.0)

    # t
    x_min = 84
    x_max = 96
    x = generateOneX(dataSize, cutoff)
    X[:,5] = 90 + x*((x_max-x_min)/cutoff/2.0)

    # Round off decimals
    X = np.around(X, decimals = 4)

    return X

#--------------------------------------------------------------------------------
# Generate Abinit input file
#--------------------------------------------------------------------------------
def saveAbinitInputFile(abcrst, filename):

    # Mg lattice parameters
    La = 5.89
    Lb = 10.201779256580686
    Lc = 9.56

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
  optcell 0
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
""" %(La*abcrst[0],Lb*abcrst[1],Lc*abcrst[2],abcrst[3],abcrst[4],abcrst[5])

    f = open(filename,'w')
    f.write(fileContent)
    f.close()

    return

#---------------------------------------------------------------------------
# Make directories for output
#---------------------------------------------------------------------------
def makeDirectories(totalBatches):
    for batchNumber in range(1, totalBatches+1):
        path='batch' + str(batchNumber) + '/Output/'
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return

#---------------------------------------------------------------------------
# Generate SBATCH scripts
#---------------------------------------------------------------------------
def generateSbatchScripts(batchNumber, minDataNumber, maxDataNumber):
    lines = """#!/bin/sh

############################################################################
# RUN ABINIT
############################################################################

#SBATCH --job-name="mg%i"
#SBATCH --ntasks=14
#SBATCH --time=48:00:00
#SBATCH --mail-user=yteh@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_mg%i.out
#SBATCH --error=error_file%i.txt

echo Working directory is: $SLURM_SUBMIT_DIR
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS = ${NPROCS}

# Load environment
module load openmpi/3.0.0
module load hdf5/1.10.1
module load netcdf-fortran/4.4.4
module load fftw/3.3.7
module load blas/3.8.0
module load lapack/3.8.0

for NUM in $(seq %i %i); do

Name=mg${NUM}

cat > mg.files << EOF
../Abinit_input/${Name}.in
Output/${Name}.out
Output/${Name}i
Output/${Name}o
Output/${Name}
../Psps/mg.pseu
EOF

srun /central/groups/bhatta/yteh/abinit-8.10.2/src/98_main/abinit < mg.files > log_abinit
done

echo Done!
""" %(batchNumber,batchNumber,batchNumber,minDataNumber,maxDataNumber)
    
    filename = 'batch' + str(batchNumber) + '/mg_batch' + \
               str(batchNumber) +'.sh'
    f = open(filename,'w')
    f.write(lines)
    f.close()

    return

#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------
def main():
    # Parameters: cutoff on normal distribution curve, number of sets of data
    cutoff = 2
    startDataNumber = 1
    dataSize = 3000
    batchSize = 100

    # Generate X and save file as mg_X.npy
    X = generateAllX(dataSize, cutoff)
    np.save('mg_X',X)

    # Make directory for Abinit input files
    dirName = 'Abinit_input/'
    try:
        os.makedirs(dirName)
    except:
        if not os.path.isdir(dirName):
            raise

    # Generate and save Abinit input files
    for i in range(dataSize):
        abcrst = X[i,:]
        filename = dirName + 'mg%i.in' %(i+startDataNumber)
        saveAbinitInputFile(abcrst,filename)

    # Create directories for output
    totalBatches = int(math.ceil(float(dataSize)/float(batchSize)))
    makeDirectories(totalBatches)

    # Generate bash scripts
    for batchNumber in range(1,totalBatches+1):
        minDataNumber = batchSize*(batchNumber-1) + startDataNumber
        maxDataNumber = min(batchSize*batchNumber+startDataNumber-1, dataSize+startDataNumber-1)
        generateSbatchScripts(batchNumber, minDataNumber, maxDataNumber)

if __name__ == '__main__': main()
