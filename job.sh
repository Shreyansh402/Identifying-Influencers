#!/bin/sh
#PBS -N COL730
#PBS -P col730.mt1210230.course
#PBS -q standard
#PBS -M $mt1210230@iitd.ac.in
#PBS -m bea
#PBS -l select=2:ncpus=40
#PBS -l walltime=06:00:00

## Environment
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd /home/maths/btech/mt1210230

## Modules
module purge
module load compiler/gcc/11.2/openmpi/4.1.6

## CPU JOB
make 
mpirun --mca opal_warn_on_missing_libcuda 0 -np 1 ./2021MT10230 42
mpirun --mca opal_warn_on_missing_libcuda 0 -np 2 ./2021MT10230 42
mpirun --mca opal_warn_on_missing_libcuda 0 -np 4 ./2021MT10230 42
mpirun --mca opal_warn_on_missing_libcuda 0 -np 8 ./2021MT10230 42
mpirun --mca opal_warn_on_missing_libcuda 0 -np 16 ./2021MT10230 42
mpirun --mca opal_warn_on_missing_libcuda 0 -np 32 ./2021MT10230 42
mpirun --mca opal_warn_on_missing_libcuda 0 -np 40 ./2021MT10230 42