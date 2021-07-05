#!/bin/sh

cd tmp

for NUM in $(seq 1 1000); do

NAME=pred${NUM}
echo $NAME

cp ../Electron_density_binary_files/${NAME}o_DEN ../Electron_density_binary_files/${NAME}o_DS1_DEN

cat > abinit.files << EOF
../AbinitInput2/${NAME}.in
../AbinitOutput2/${NAME}.out
tmpi
../Electron_density_binary_files/${NAME}o
tmp
../../00_AbinitRun/Psps/mg.pseu
EOF

mpirun -np 18 abinit < abinit.files > log_abinit

mv ../Electron_density_binary_files/${NAME}o_DS2_VXC.nc ../AbinitOutput2/${NAME}_VXC.nc

done

