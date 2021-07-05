#!/bin/sh

for NUM in $(seq 1 38); do
sbatch run_batch${NUM}.sh
sleep 2 #Delay for 5 seconds
done
