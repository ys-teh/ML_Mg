#/usr/bin/env python3

#################################################################################
## Generate X data, i.e. a,b,c,r,s,t, following normal distribution
## At zero deformation, a = b = c = 1, r = s = t = 90deg
#################################################################################

import numpy as np
import math
import os, glob
import netCDF4
from subprocess import check_call

#--------------------------------------------------------------------------------
# Run cut3d for all bands for each k point
#--------------------------------------------------------------------------------
def run_cut3d(ikpt,nband):
    # Create cut3d.file
    content = 'Output/mgo_WFK.nc\n' + str(ikpt+1) + '\n1\n0\n0\n1\ntmp/mg\n'
    for iband in range(2,nband+1):
        content = content + '1\n' + str(ikpt+1) + '\n' + str(iband) + '\n0\n0\n1\ntmp/mg\n'
    content = content + '0'

    f = open('tmp/cut3d.file','w+')
    f.write(content)
    f.close()

    # Run cut3d < cut3d.file > log_cut3d
    with open('tmp/cut3d.file','r') as input_pipe:
        with open('tmp/log_cut3d','w') as output_pipe:
            proc = check_call(['cut3d'], stdin=input_pipe, stdout=output_pipe)

    return

#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------
def main():

    # Read occupation number (occ), eigenvalues (eig) and k point weight (wtk)
    # from NetCDF file
    nc = netCDF4.Dataset('Output/mgo_WFK.nc','r')
    print('Shape of occ: ', nc.variables['occupations'].shape)
    occ = nc.variables['occupations'][:]
    print('Shape of eig: ', nc.variables['eigenvalues'].shape)
    eig = nc.variables['eigenvalues'][:]
    wtk_shape = nc.variables['kpoint_weights'].shape
    print('Shape of wtk: ', wtk_shape)
    wtk = nc.variables['kpoint_weights'][:]
    nc.close()

    # Variables from Abinit
    nfft = 36*64*60
    nkpt = wtk_shape[0]
    nband = 20
  
    # Create tmp directory
    try:
        os.makedirs('tmp/')
    except:
        if not os.path.isdir('tmp/'):
            raise
  
    # Compute band structure energy (U) and electronic entropy (S)
    U = np.zeros((nfft))
    S = np.zeros((nfft))
    for ikpt in range(nkpt):
        print('ikpt no: ', ikpt)
        run_cut3d(ikpt,nband)
        arr1 = np.zeros((nfft))
        arr2 = np.zeros((nfft))
        for iband in range(nband):
            filename = 'tmp/mg_k' + str(ikpt+1) + '_b' + str(iband+1)
            psi = np.loadtxt(filename, dtype=np.float64)
            psiSquare = np.square(psi[:,0]) + np.square(psi[:,1])
            occKbd = occ[0,ikpt,iband]
#            occKbdHalf = occKbd/2.0
            arr1 += (eig[0,ikpt,iband] * occKbd) * psiSquare
#            sca = occKbd*math.log(occKbdHalf) + (2.0-occKbd)*math.log(1.0 - occKbdHalf)
#            arr2 += sca * psiSquare
        U += arr1 * wtk[ikpt]
#        S -= arr2 * wtk[ikpt]

        #Delete files to save space
        allFiles = glob.glob('tmp/*')
        for i in allFiles:
            os.remove(i)

    # Check values 
    print('Total band energy: ',np.sum(U)/float(36*64*60),'Hartree')
#    print('Total electronic entropy',-np.sum(S)*0.02/float(36*64*60),'Hartree')

    # Save files
    np.savetxt('band_energy.txt',U)
#    np.savetxt('electronic_entropy.txt',S)

if __name__ == '__main__': main()
