#!/bin/sh

cd tmp

for NUM in $(seq 661 1000); do

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

