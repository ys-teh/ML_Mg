#!/bin/sh

for NUM in $(seq 1 10); do
cd batch${NUM}
sbatch mg_batch${NUM}.sh
cd ..
sleep 2 #Delay for 2 seconds
done
