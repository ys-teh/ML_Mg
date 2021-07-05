#/usr/bin/env python3

#################################################################################
## Save data (DEN, VCLMB, XRED, BAND, ENTR) in the form of .npy
## Created by Ying Shi Teh
## Last modified on July 5, 2021
#################################################################################

import numpy as np
import os, glob
import time
import math
from subprocess import check_call, CalledProcessError
import shutil
import netCDF4

##-------------------------------------------------------------------------------
## Load all DEN
##-------------------------------------------------------------------------------
def load_normalize_all_electron_density(acell,nfft):
    allStrainVals = np.load('mg_X.npy')
    dataSize = np.shape(allStrainVals)[0]
    allDensity = np.empty((dataSize,nfft))
    strain = np.empty((3,3))
    identity = np.identity(3)
    for index in range(0,dataSize):
        # Load electron density
        filename = 'Abinit_output/mg' + str(index+1) + '_DEN.nc'
        with netCDF4.Dataset(filename,'r') as nc:
            density = nc.variables['density'][:]
        density = np.reshape(density,(nfft))

        # Normalize electron density
        strainVals = allStrainVals[index]
        a = strainVals[0]
        b = strainVals[1]
        c = strainVals[2]
        r = strainVals[3]
        s = strainVals[4]
        t = strainVals[5]
        cos_r = math.cos(math.radians(r))
        cos_s = math.cos(math.radians(s))
        cos_t = math.cos(math.radians(t))
        dV = (acell[0]*acell[1]*acell[2]*a*b*c)/nfft \
             * math.sqrt( 1 + 2*cos_r*cos_s*cos_t - cos_r*cos_r - cos_s*cos_s - cos_t*cos_t )
        density *= dV

        # Check that the sum of electrons is 8.0
        if abs(np.sum(density)-8.0)>1e-6:
            print("Error: Sum of electrons is %f!" %(np.sum(density)))
#        else:
#            print("Sum of electrons is %f!" %(np.sum(density)))

        # Save density
        allDensity[index,:] = density

    return allDensity, dataSize

##-------------------------------------------------------------------------------
## Load all VCLMB
##-------------------------------------------------------------------------------
def load_all_potential(nfft,dataSize):
    allPotential = np.empty((dataSize,nfft))
    time0 = time.time()
    for index in range(0,dataSize):
        # Load potential
        filename = 'Abinit_output/mg' + str(index+1) + '_VCLMB.nc'
        with netCDF4.Dataset(filename,'r') as nc:
            potential = nc.variables['vhartree_vloc'][:]
        potential = np.reshape(potential,(nfft))

        # Save potential
        allPotential[index,:] = potential

    return allPotential

##-------------------------------------------------------------------------------
## Load all VHA
##-------------------------------------------------------------------------------
def load_all_hartree_potential(nfft,dataSize):
    allPotential = np.empty((dataSize,nfft))
    time0 = time.time()
    for index in range(0,dataSize):
        # Load potential
        filename = 'Abinit_output/mg' + str(index+1) + '_VHA.nc'
        with netCDF4.Dataset(filename,'r') as nc:
            potential = nc.variables['vhartree'][:]
        potential = np.reshape(potential,(nfft))

        # Save potential
        allPotential[index,:] = potential

    return allPotential


##-------------------------------------------------------------------------------
## Get XRED (atomic positions)
##-------------------------------------------------------------------------------
def get_xred(filename):
    # Get xred
    xred = np.empty((3))
    countLine = 0
    with open(filename) as inputFile:
        flag = False
        for line in inputFile:
            if 'after computation' in line:
                flag = True
                continue
            if 'xred' in line and flag:
                countLine = 1
                tmp = line.split()
                xred[0] = tmp[1]
                xred[1] = tmp[2]
                xred[2] = tmp[3]
#            elif flag and countLine is 1:
#                countLine = 2
#                tmp = line.split()
#                xred[3] = tmp[0]
#                xred[4] = tmp[1]
#                xred[5] = tmp[2]
#            elif flag and countLine is 2:
#                countLine = 3
#                tmp = line.split()
#                xred[6] = tmp[0]
#                xred[7] = tmp[1]
#                xred[8] = tmp[2]
#            elif flag and countLine is 3:
#                countLine = 4
#                tmp = line.split()
#                xred[9] = tmp[0]
#                xred[10] = tmp[1]
#                xred[11] = tmp[2]
                break

    return xred

##-------------------------------------------------------------------------------
## Load all XRED
##-------------------------------------------------------------------------------
def load_all_xred(dataSize):
    allXred = np.empty((dataSize,3))
    for index in range(0,dataSize):
        # Go through Abinit output files
        filename = 'Abinit_output/mg' + str(index+1) + '.out'
        xred = get_xred(filename)

        # Save potential
        allXred[index,:] = xred

    return allXred

##-------------------------------------------------------------------------------
## Get RPRIM
##-------------------------------------------------------------------------------
def get_rprim(filename):
    # Get rprim
    rprim = np.empty((9))
    countLine = 0
    with open(filename) as inputFile:
        for line in inputFile:
            if 'rprim' in line:
                countLine = 1
                tmp = line.split()
                rprim[0] = tmp[1]
                rprim[1] = tmp[2]
                rprim[2] = tmp[3]
                continue
            elif countLine is 1:
                countLine = 2
                tmp = line.split()
                rprim[3] = tmp[0]
                rprim[4] = tmp[1]
                rprim[5] = tmp[2]
                continue
            elif countLine is 2:
                countLine = 3
                tmp = line.split()
                rprim[6] = tmp[0]
                rprim[7] = tmp[1]
                rprim[8] = tmp[2]
                break

    return rprim

##-------------------------------------------------------------------------------
## Load all RPRIM
##-------------------------------------------------------------------------------
def load_all_rprim(dataSize):
    allRprim = np.empty((dataSize,9))
    for index in range(0,dataSize):
        # Go through Abinit output files
        filename = 'Abinit_output/mg' + str(index+1) + '.out'
        rprim = get_rprim(filename)

        # Save potential
        allRprim[index,:] = rprim

    return allRprim

##-------------------------------------------------------------------------------
## Initialize inverse coordinates
##-------------------------------------------------------------------------------
def get_inverse_coordinates(ngfft,istwf_k):
    n1 = ngfft[0]
    n2 = ngfft[1]
    n3 = ngfft[2]
    i1inv = np.zeros((n1),dtype=int)
    i2inv = np.zeros((n2),dtype=int)
    i3inv = np.zeros((n3),dtype=int)

    if (istwf_k==2) or (istwf_k==4) or (istwf_k==6) or (istwf_k==8):
        for i1 in range(1,n1):
            i1inv[i1] = n1 - i1
    else:
        for i1 in range(n1):
            i1inv[i1] = n1 - i1 - 1
    if (istwf_k>=2) and (istwf_k<=5):
        for i2 in range(1,n2):
            i2inv[i2] = n2 - i2
    else:
        for i2 in range(n2):
            i2inv[i2] = n2 - i2 -1
    if (istwf_k==2) or (istwf_k==3) or (istwf_k==6) or (istwf_k==7):
        for i3 in range(1,n3):
            i3inv[i3] = n3 - i3
    else:
        for i3 in range(n3):
            i3inv[i3] = n3 - i3 - 1

    return i1inv, i2inv, i3inv

##-------------------------------------------------------------------------------
## Convert wavefunction coefficients from sphere to real (istwf_k=1)
## Reference: abinit-8.10.3/src/52_fft_mpi_noabirule/m_fftcore.F90/sphere
##-------------------------------------------------------------------------------
def convert_cg_to_real_space1(ngfft,nfft,cg_k,kg_k,npw_k):
    cfft = np.zeros((ngfft[0],ngfft[1],ngfft[2]),dtype=complex)
    for ipw in range(npw_k):
        i1 = kg_k[ipw,0]
        i2 = kg_k[ipw,1]
        i3 = kg_k[ipw,2]
        cfft[i1,i2,i3] = cg_k[ipw,0] + cg_k[ipw,1]*1.0j

    # Inverse FFT: obtain wavefunction coefficients in real space
    cr_k = np.fft.ifftn(cfft)

    # Reshape from 3D to 1D
    cr_k = np.reshape(cr_k,(nfft),'F')

    return cr_k

##-------------------------------------------------------------------------------
## Convert wavefunction coefficients from sphere to real (istwf_k>=2)
## Reference: abinit-8.10.3/src/52_fft_mpi_noabirule/m_fftcore.F90/sphere
##-------------------------------------------------------------------------------
def convert_cg_to_real_space2(ngfft,nfft,cg_k,kg_k,npw_k,istwf_k,i1inv,i2inv,i3inv):
    cfft = np.zeros((ngfft[0],ngfft[1],ngfft[2]),dtype=complex)

    npwmin = 0
    if istwf_k==2:
        npwmin = 1
        cfft[0,0,0] = cg_k[0,0] # Zero imaginary component
    for ipw in range(npwmin,npw_k):
        i1 = kg_k[ipw,0]
        i2 = kg_k[ipw,1]
        i3 = kg_k[ipw,2]
        cfft[i1,i2,i3] = cg_k[ipw,0] + cg_k[ipw,1]*1.0j
        cfft[i1inv[i1],i2inv[i2],i3inv[i3]] = cg_k[ipw,0] - cg_k[ipw,1]*1.0j

    # Inverse FFT: obtain wavefunction coefficients in real space
    if istwf_k==2:
        cfft = cfft[:,:,:(ngfft[2]//2+1)]
        cr_k = np.fft.irfftn(cfft)
    else:
        cr_k = np.fft.ifftn(cfft)

    # Reshape from 3D to 1D
    cr_k = np.reshape(cr_k,(nfft),'F')

    return cr_k

##-------------------------------------------------------------------------------
## Load all EBAND, ENTR (i.e. band structure energy and electronic entropy)
##-------------------------------------------------------------------------------
def load_all_eband_and_entr(ngfft,nfft,dataSize,nband,sigma):
    allEband = np.zeros((dataSize,nfft))
    allEntr = np.zeros((dataSize,nfft))

    for idata in range(0,dataSize):
        print('data no: ', idata)

        # Read occupation number (occ), eigenvalues (eig) and k point weight (wtk)
        # from NetCDF file
        filename = 'Abinit_output/mg' + str(idata+1) + '_WFK.nc'
        with netCDF4.Dataset(filename,'r') as nc:
            occ = nc.variables['occupations'][:]
            eig = nc.variables['eigenvalues'][:]
            wtk = nc.variables['kpoint_weights'][:]
            npw = nc.variables['number_of_coefficients'][:]
            kg = nc.variables['reduced_coordinates_of_plane_waves'][:]
            cg = nc.variables['coefficients_of_wavefunctions'][:]
            istwf = nc.variables['istwfk'][:]
            fermie = nc.variables['fermi_energy'][:].data
        nkpt = np.shape(wtk)[0]

        # Compute band structure energy and electronic entropy
        for ikpt in range(nkpt):
            npw_k = npw[ikpt]
            kg_k = kg[ikpt,:npw_k,:]
            istwf_k = istwf[ikpt]
            wt_k = wtk[ikpt]

            # Convert kg_k to positive
            kg_k = kg_k + np.multiply((kg_k<0),np.array([ngfft[0],ngfft[1],ngfft[2]]))

            # Initialize inverse coordinates
            if istwf_k>=2:
                i1inv, i2inv, i3inv = get_inverse_coordinates(ngfft,istwf_k)

            # Loop over all bands
            for iband in range(nband):
                # Wavefunction coefficients in Fourier basis in sphere
                cg_k = cg[0,ikpt,iband,0,:npw_k,:]

                # Occupation number
                occKbd = occ[0,ikpt,iband]

                # Convert wavefunction coefficients to real space
                # These coefficients need to be multiplied by nfft to match with those from cut3d
                if istwf_k==1:
                    cr_k = convert_cg_to_real_space1(ngfft,nfft,cg_k,kg_k,npw_k)
                else:
                    cr_k = convert_cg_to_real_space2(ngfft,nfft,cg_k,kg_k,npw_k,istwf_k,i1inv,i2inv,i3inv)

                # Note the multiplication with nfft --> add the correct factor so that sum(psiSquare)=1.0
                psiSquare = np.square(cr_k.real) + np.square(cr_k.imag)
                allEband[idata,:] += (eig[0,ikpt,iband] * occKbd * wt_k * nfft) * psiSquare
                a = -0.5634
                x = (fermie - eig[0,ikpt,iband])/sigma
                sca = (1./math.sqrt(math.pi)) * math.exp(-x*x) * (x*x*(-a*x+1) - 0.5)
                #occKbdHalf = occKbd/2.0
                #sca = occKbd*math.log(occKbdHalf) + (2.0-occKbd)*math.log(1.0 - occKbdHalf)
                allEntr[idata,:] += (-sca * wt_k * nfft) * psiSquare

    return allEband, allEntr

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():
    time0 = time.time()

    # Variables from Abinit
    ngfft = [36,64,60]
    nfft = ngfft[0]*ngfft[1]*ngfft[2]
    acell = [5.89,10.201779256580686,9.56]
    nband = 20
    sigma = 0.01 #Cold smearing broadening

    # Data (electron density)
    print("Load and normalize density")
    Y_DEN, dataSize = load_normalize_all_electron_density(acell,nfft)
    np.save('mg_Y_DEN.npy',Y_DEN)

    # Data (potential)
    print("Load potential")
    Y_VCLMB = load_all_potential(nfft,dataSize)
    np.save('mg_Y_VCLMB.npy',Y_VCLMB)

    # Data (potential)
    print("Load potential")
    Y_VHA = load_all_hartree_potential(nfft,dataSize)
    np.save('mg_Y_VHA.npy',Y_VHA)

    # Data (atomic position)
    print("Load xred")
    Y_XRED = load_all_xred(dataSize)
    np.save('mg_Y_XRED.npy',Y_XRED)

    # Data (rprim)
    print("Load rprim")
    X_RPRIM = load_all_rprim(dataSize)
    np.save('mg_RPRIM.npy',X_RPRIM)

#    # Data (band structure energy, electronic entropy)
#    print("Load eband and entr")
#    dataSize = 50
#    Y_EBAND, Y_ENTR = load_all_eband_and_entr(ngfft,nfft,dataSize,nband,sigma)
#    np.save('mg_Y_EBAND.npy',Y_EBAND)
#    np.save('mg_Y_ENTR.npy',Y_ENTR)
#    # Check total band energy and electronic entropy (both in Hartree)
#    total_eband = np.sum(Y_EBAND,axis=1)
#    total_entr = -np.sum(Y_ENTR,axis=1)*sigma
#    np.savetxt('check_total_band_energy.txt', total_eband)
#    np.savetxt('check_total_entropy.txt', total_entr)
#
    # Computation time
    print("Total computation time: %f" %(time.time()-time0))

if __name__ == '__main__': main()

