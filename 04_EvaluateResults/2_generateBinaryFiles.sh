#!/bin/sh

############################################################################
# GENERATE BINARY FILES FOR ELECTRON DENSITY
# Create by Ying Shi Teh
# Last modified on July 5, 2021
############################################################################
if [ ! -d "tmp/" ]; then
    mkdir tmp
fi

#if [ -d "Electron_density_binary_files/" ]; then
#    rm -r Electron_density_binary_files
#fi
#mkdir Electron_density_binary_files/

cd tmp

for NUM in $(seq 423 1000)
do

#---------------------------------------------------------------------------
# Generate Abinit input files (arbitrary) to generate binary files
#---------------------------------------------------------------------------

NAME=pred$NUM
echo $NAME

cat > tmp.in << EOF
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
  acell 6.04 10.5 9.8
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
  nstep 0          # Maximal number of SCF cycles
  toldfe 1.0d-10

EOF

#---------------------------------------------------------------------------
# Renaming electron density files
#---------------------------------------------------------------------------
FILE=../Electron_density_files/${NAME}.dat
if [ ! -f $FILE ]; then
    echo "File does not exist!"
    break
fi
cp ${FILE} /home/yteh/abinit-8.10.2/density_file_added/density_file.dat

#---------------------------------------------------------------------------
# Run Abinit (using modified code for src/94_scfcv/m_outscfcv.f90)
#---------------------------------------------------------------------------

cat > tmp_abinit.files << EOF
tmp.in
tmp.out
tmpi
../Electron_density_binary_files/${NAME}o
tmp
../../00_AbinitRun/Psps/mg.pseu
EOF

mpirun -np 18 /home/yteh/abinit-8.10.2/src/98_main/abinit < tmp_abinit.files > log_abinit

done

cd ..
echo Task completed!
