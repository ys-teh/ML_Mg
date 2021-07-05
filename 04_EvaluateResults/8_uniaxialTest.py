#!/usr/bin/env python3

################################################################################
# Generate stress-strain curve from machine learning results
# Create by Ying Shi Teh
# Last modified on July 5, 2021
################################################################################

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
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#-------------------------------------------------------------------------------
# Remove file
#-------------------------------------------------------------------------------
def remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

    return

#-------------------------------------------------------------------------------
# Compute total free energy from strain
#-------------------------------------------------------------------------------
def compute_total_energy(strain,saveDir,saveDirXRED,nfft,acell,tsmear):

    DEN,EBAND,ENTR,XRED = generate_predictions(strain,saveDir,saveDirXRED)
    generate_DEN_binary_file(DEN,strain,nfft,acell)
    run_abinit(XRED,strain,acell)
    totalEnergy = extract_and_compute_total_energy(DEN,EBAND,ENTR,tsmear)

    return totalEnergy

#-------------------------------------------------------------------------------
# Generate DEN, EBAND, ENTR, XRED predictions given strain
# Input: strain (e.g. np.array([0.0,0.0,0.0]))
#-------------------------------------------------------------------------------
def generate_predictions(strain,saveDir,saveDirXRED):

    # Deformation parameters
    X = np.array([[1.0,1.0,1.0,90,90,90]])
    X[0,0:3] += strain

    # Predict DEN
    mlDir = saveDir + 'DEN/'
    scalerX = joblib.load(mlDir+'scalerX')
    scalerY = joblib.load(mlDir+'scalerY')
    pca = joblib.load(mlDir+'pca')
    model = keras.models.load_model(mlDir+'ml_model.h5')
    Y = np.copy(X)
    Y = scalerX.transform(Y)
    Y = model.predict(Y)
    Y = scalerY.inverse_transform(Y)
    Y = pca.inverse_transform(Y)
    DEN = np.copy(Y[0])

    # Predict EBAND
    mlDir = saveDir + 'EBAND/'
    scalerX = joblib.load(mlDir+'scalerX')
    scalerY = joblib.load(mlDir+'scalerY')
    pca = joblib.load(mlDir+'pca')
    model = keras.models.load_model(mlDir+'ml_model.h5')
    Y = np.copy(X)
    Y = scalerX.transform(Y)
    Y = model.predict(Y)
    Y = scalerY.inverse_transform(Y)
    Y = pca.inverse_transform(Y)
    EBAND = np.copy(Y[0])

    # Predict ENTR
    mlDir = saveDir + 'ENTR/'
    scalerX = joblib.load(mlDir+'scalerX')
    scalerY = joblib.load(mlDir+'scalerY')
    pca = joblib.load(mlDir+'pca')
    model = keras.models.load_model(mlDir+'ml_model.h5')
    Y = np.copy(X)
    Y = scalerX.transform(Y)
    Y = model.predict(Y)
    Y = scalerY.inverse_transform(Y)
    Y = pca.inverse_transform(Y)
    ENTR = np.copy(Y[0])

    # Predict XRED
    scalerX = joblib.load(saveDirXRED+'scalerX')
    scalerY = joblib.load(saveDirXRED+'scalerY')
    model = keras.models.load_model(saveDirXRED+'ml_model.h5')
    Y = np.copy(X)
    Y = scalerX.transform(Y)
    Y = model.predict(Y)
    Y = scalerY.inverse_transform(Y)
    XRED = np.copy(Y[0])

    return DEN,EBAND,ENTR,XRED

#-------------------------------------------------------------------------------
# Generate binary file for DEN
#-------------------------------------------------------------------------------
def generate_DEN_binary_file(DEN,strain,nfft,acell):

    # Unnormalize and save electron density
    dV = acell[0]*acell[1]*acell[2]*(1.0+strain[0])*(1.0+strain[1])*(1.0+strain[2])/nfft
    density = DEN/dV
    filename = '/home/yteh/abinit-8.10.2/density_file_added/density_file.dat'
    remove(filename)
    np.savetxt(filename,density)

    # Save abinit input file
    fileContent = """
  # Magnesium: Structural Optimization

  # Output parameters
  prtwf 0
  prtden 1
  prteig 0
  prtebands 0

  # Occupation options
  occopt 4
  tsmear 0.01
  nband 20

  #Definition of the unit cell
  acell 5.89 10.201779256580686 9.56
  angdeg 90 90 90
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
  nstep 0          # Maximal number of SCF cycles
  toldfe 1.0d-10
""" 
    f = open('tmp/tmp.in','w')
    f.write(fileContent)
    f.close()

    # Save abinit .files file
    fileContent = """tmp.in
tmp.out
tmpi
tmpo
tmp
../../00_AbinitRun/Psps/mg.pseu
"""
    f = open('tmp/tmp_abinit.files','w')
    f.write(fileContent)
    f.close()

    # Run Abinit
    runLine = 'cd tmp; mpirun -np 18 /home/yteh/abinit-8.10.2/src/98_main/abinit \
               < tmp_abinit.files > log_abinit; cd ..'
    os.system(runLine)

    return

#-------------------------------------------------------------------------------
# Run Abinit using DEN binary file generated
#-------------------------------------------------------------------------------
def run_abinit(XRED,strain,acell):

    # Reduced coordinates of atoms
    xredLines = """%.6f %.6f %.6f
  %.6f %.6f %.6f
  %.6f %.6f %.6f
  %.6f %.6f %.6f
"""%(XRED[0], XRED[1], XRED[2], \
     0.5+XRED[0], 0.5+XRED[1], XRED[2], \
     -XRED[0], 2.0/3.0-XRED[1], 0.5-XRED[2], \
     0.5-XRED[0], 1.0/6.0-XRED[1], 0.5-XRED[2])

    # Abinit input file
    fileContent = """
  # Magnesium: Structural Optimization

  ndtset 2

  # Output parameters 
  prtwf 0
  prtden 0
  prteig 0
  prtvxc2 1
  getden2 1
  iomode 3

  # Occupation options 
  occopt 4
  tsmear 0.01
  nband 20

  #Definition of the unit cell
  acell %.12f %.12f %.12f
  angdeg 90 90 90
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
  nstep 0
  toldfe 1.0d-10
"""%(acell[0]*(1.0+strain[0]),acell[1]*(1.0+strain[1]),acell[2]*(1.0+strain[2]),xredLines)

    f = open('tmp/tmp.in','w')
    f.write(fileContent)
    f.close()

    # Save abinit .files file
    fileContent = """tmp.in
tmp.out
tmpi
tmpo
tmp
../../00_AbinitRun/Psps/mg.pseu
"""
    f = open('tmp/tmp_abinit.files','w')
    f.write(fileContent)
    f.close()

    # Run Abinit
    os.rename('tmp/tmpo_DEN','tmp/tmpo_DS1_DEN')
    remove('tmp/tmp.out')
    remove('tmp/tmpo_DS2_VXC.nc')
    runLine = 'cd tmp; mpirun -np 18 abinit < tmp_abinit.files > log_abinit; cd ..'
    os.system(runLine)

    return 

#-------------------------------------------------------------------------------
# Extract relevant energy components from Abinit file and compute total energy
#-------------------------------------------------------------------------------
def extract_and_compute_total_energy(DEN,EBAND,ENTR,tsmear):

    # Extract energy values
    flag = False
    with open('tmp/tmp.out') as input_file:
        for line in input_file:
            if '== DATASET  2 ==' in line:
                flag = True
            if flag:
                if 'Hartree energy' in line:
                    hartreeEnergy = line
                    continue
                elif 'XC energy' in line:
                    xcEnergy = line
                    continue
                elif 'Ewald energy' in line:
                    ewaldEnergy = line
                    continue
                elif 'PspCore energy' in line:
                    pspcoreEnergy = line
                    break
    hartreeEnergy = float(hartreeEnergy.split()[3])
    xcEnergy = float(xcEnergy.split()[3])
    ewaldEnergy = float(ewaldEnergy.split()[3])
    pspcoreEnergy = float(pspcoreEnergy.split()[3])

    # Compute \int Vxc(r)*rho(r)dr
    with netCDF4.Dataset('tmp/tmpo_DS2_VXC.nc','r') as nc:
        Vxc = nc.variables['exchange_correlation_potential'][:]
        Vxc = Vxc.reshape((np.size(Vxc)))
    VxcDEN = np.sum(np.multiply(DEN,Vxc))

    # Total energy
    intEnergy = np.sum(EBAND) - VxcDEN - hartreeEnergy + xcEnergy + ewaldEnergy + pspcoreEnergy
    totalEnergy = intEnergy + np.sum(ENTR)*(-tsmear)
 
    return totalEnergy

#-------------------------------------------------------------------------------
# Compute energy gradient
#-------------------------------------------------------------------------------
def compute_energy_gradient(strain,saveDir,saveDirXRED,nfft,acell,tsmear):

    energy = compute_total_energy(strain,saveDir,saveDirXRED,nfft,acell,tsmear)

    strainStep = 1e-4
    energyGradient = np.zeros((3))

    for strainIndex in range(3):
        strain2 = np.copy(strain)
        strain2[strainIndex] += strainStep
        energy2 = compute_total_energy(strain2,saveDir,saveDirXRED,nfft,acell,tsmear)
        energyGradient[strainIndex] = (energy2-energy)/strainStep

    return energyGradient

#-------------------------------------------------------------------------------
# Gradient descent
#-------------------------------------------------------------------------------
def gradient_descent(strain,fixedStrainIndex,saveDir,saveDirXRED,nfft,acell,tsmear):

    nIterations = 30
    gamma = 0.5
    tol = 5e-5
    relax = np.array([1.,1.,1.])
    relax[fixedStrainIndex] = 0.
    for iter in range(nIterations):
        print('Iteration: ',iter)
        energyGradient = compute_energy_gradient(strain,saveDir,saveDirXRED,nfft,acell,tsmear)

        # Stopping criterion
        if np.max(np.absolute(relax*energyGradient)) < tol:
            print('Convergence has reached at iteration ',iter)
            break
        elif (iter == (nIterations-1)):
            print('Solution is not converged!')

        # Standard gradient descent
        if iter is 0:
            energyGradientOld = np.copy(energyGradient)
        # BB method
        else:
            energyGradientDiff = relax*(energyGradient - energyGradientOld)
            denominator = np.sum(np.square(energyGradientDiff))
            if denominator > 0:
                gamma = np.absolute(np.sum(np.multiply(deltaStrain,energyGradientDiff)))/denominator
            energyGradientOld = np.copy(energyGradient)

        # Update x
        deltaStrain = -gamma*relax*energyGradient
        strain += deltaStrain
        print('Energy gradient: ',energyGradient)
        print('Strain: ',strain)

    return strain,energyGradient

#-------------------------------------------------------------------------------
# Stress-strain
#-------------------------------------------------------------------------------
def plot_stress_strain(fixedStrainIndex,saveDir,saveDirXRED,nfft,acell,tsmear):

    N = 61
    uniaxialStrain = np.linspace(-0.1,0.1,N)
    stress = np.zeros((N))
    allStrain = np.zeros((N,3))

    strain = np.array([0.0,0.0,0.0])
    for i in range(math.floor(N/2),N):
        strain[fixedStrainIndex] = uniaxialStrain[i]
        strain,energyGradient = gradient_descent(strain,fixedStrainIndex,saveDir,saveDirXRED,nfft,acell,tsmear)
        vol = acell[0]*acell[1]*acell[2]*(strain[0]+1)*(strain[1]+1.0)*(strain[2]+1.0)
        stress[i] = energyGradient[fixedStrainIndex]/vol
        allStrain[i] = strain
    strain = np.array([0.0,0.0,0.0])
    for i in range(math.floor(N/2)-1,-1,-1):
        strain[fixedStrainIndex] = uniaxialStrain[i]
        strain,energyGradient = gradient_descent(strain,fixedStrainIndex,saveDir,saveDirXRED,nfft,acell,tsmear)
        vol = acell[0]*acell[1]*acell[2]*(strain[0]+1)*(strain[1]+1.0)*(strain[2]+1.0)
        stress[i] = energyGradient[fixedStrainIndex]/vol
        allStrain[i] = strain
    np.save('Uniaxial_test_results/stress_direction'+str(fixedStrainIndex+1)+'.npy',stress)
    np.save('Uniaxial_test_results/strain_direction'+str(fixedStrainIndex+1)+'.npy',allStrain)

    # Load abinit stress values
    strainAbinit = np.linspace(-0.1,0.1,21)
    stressAbinit = np.loadtxt('../06_ML_ETOT/Abinit_runs/direction'+str(fixedStrainIndex+1)+'/mg_summarizedStressValues.txt')

    markerType = ['o','x','^']
    label1 = ['ML (1)','ML (2)','ML (3)']
    label2 = ['True (1)','True (2)','True (3)']
    if fixedStrainIndex is 0:
        plt.figure(figsize=(5,3.5))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.plot(uniaxialStrain,stress*1000,label=label1[fixedStrainIndex],color='C0')
    plt.plot(strainAbinit,stressAbinit*1000,markerType[fixedStrainIndex],label=label2[fixedStrainIndex],color='C0',mfc='none')
    plt.xlabel('Strain')
    plt.ylabel('Stress (mHartree/Bohr$^3$)')

    if fixedStrainIndex is 2:
        plt.legend(frameon=False,labelspacing=0.2)
        #plt.show()
        plt.savefig('Figures/stress_strain2.pdf', bbox_inches='tight')

    return


#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def main():

    # Some parameters    
    saveDir = '../01_ML_fields/Saved_ml_results_occopt4_train2000/'
    saveDirXRED = '../02_ML_XRED/Saved_ml_results_occopt4_train2000/XRED/'
    tsmear = 0.01
    acell = [5.89,10.201779256580686,9.56]
    ngfft = [36,64,60]
    nfft = ngfft[0]*ngfft[1]*ngfft[2]

    for fixedStrainIndex in range(0,1):
        plot_stress_strain(fixedStrainIndex,saveDir,saveDirXRED,nfft,acell,tsmear)



if __name__ == '__main__': main()
