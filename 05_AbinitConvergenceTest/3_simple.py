#/usr/bin/env python3

#################################################################################
## Save data (DEN, VCLMB, XRED, BAND, ENTR) in the form of .npy
#################################################################################

import numpy as np
import os, glob
import time
import math
from subprocess import check_call, CalledProcessError
import shutil
import netCDF4

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
def load_all_eband_and_entr(ngfft,nfft,nband,sigma):
    allEband = 0.0
    allEntr = 0.0

    # Read occupation number (occ), eigenvalues (eig) and k point weight (wtk)
    # from NetCDF file
    filename = 'Output/mgo_WFK.nc'
    with netCDF4.Dataset(filename,'r') as nc:
        occ = nc.variables['occupations'][:]
        eig = nc.variables['eigenvalues'][:]
        wtk = nc.variables['kpoint_weights'][:]
        npw = nc.variables['number_of_coefficients'][:]
#        kg = nc.variables['reduced_coordinates_of_plane_waves'][:]
#        cg = nc.variables['coefficients_of_wavefunctions'][:]
        istwf = nc.variables['istwfk'][:]
        fermie = nc.variables['fermi_energy'][:].data
    nkpt = np.shape(wtk)[0]

    print('Check weight: ',np.sum(wtk))

    # Compute band structure energy and electronic entropy
    for ikpt in range(nkpt):
        print('ikpt: ',ikpt,'out of ',nkpt)

        npw_k = npw[ikpt]
#        kg_k = kg[ikpt,:npw_k,:]
        istwf_k = istwf[ikpt]
        wt_k = wtk[ikpt]

        # Convert kg_k to positive
#        kg_k = kg_k + np.multiply((kg_k<0),np.array([ngfft[0],ngfft[1],ngfft[2]]))

        # Initialize inverse coordinates
        if istwf_k>=2:
            i1inv, i2inv, i3inv = get_inverse_coordinates(ngfft,istwf_k)

        # Loop over all bands
        for iband in range(nband):
            # Wavefunction coefficients in Fourier basis in sphere
#            cg_k = cg[0,ikpt,iband,0,:npw_k,:]

            # Occupation number
            occKbd = occ[0,ikpt,iband]

            # Convert wavefunction coefficients to real space
            # These coefficients need to be multiplied by nfft to match with those from cut3d
#            if istwf_k==1:
#                cr_k = convert_cg_to_real_space1(ngfft,nfft,cg_k,kg_k,npw_k)
#            else:
#                cr_k = convert_cg_to_real_space2(ngfft,nfft,cg_k,kg_k,npw_k,istwf_k,i1inv,i2inv,i3inv)

            # Note the multiplication with nfft --> add the correct factor so that sum(psiSquare)=1.0
#            psiSquare = np.square(cr_k.real) + np.square(cr_k.imag)
            allEband += (eig[0,ikpt,iband] * occKbd * wt_k) # * nfft)

            # Cold smearing
            x = (fermie - eig[0,ikpt,iband])/sigma
            a = -0.5634
            #a = -0.8165
            #a = 0
            sca = (1./math.sqrt(math.pi)) * math.exp(-x*x) * (x*x*(-a*x+1) - 0.5)
            # Gaussian
            #sca = (1./math.sqrt(math.pi)) * math.exp(-x*x)
            allEntr += (-sca * wt_k) # * nfft)

    return allEband, allEntr

##-------------------------------------------------------------------------------
## Main
##-------------------------------------------------------------------------------
def main():

    # Variables from Abinit
    ngfft = [36,64,60]
    nfft = ngfft[0]*ngfft[1]*ngfft[2]
    acell = [6.0419031073,10.464702002,9.8453372801]
    nband = 20
    sigma = 0.01

    # Data (band structure energy, electronic entropy)
    print("Load eband and entr")
    Y_EBAND, Y_ENTR = load_all_eband_and_entr(ngfft,nfft,nband,sigma)
    # Check total band energy and electronic entropy (both in Hartree)
    total_eband = Y_EBAND
    total_entr = -Y_ENTR*sigma
    print('Total band energy: ', total_eband)
    print('Total entropy: ', total_entr)

if __name__ == '__main__': main()
