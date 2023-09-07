#!/bin/bash
#PBS -N FIDRun
#PBS -q serial
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -P CSCI1142
#PBS -l walltime=4:00:00
#PBS -o /mnt/lustre/users/schetty1/InceptionRuns/FID.out
#PBS -e /mnt/lustre/users/schetty1/InceptionRuns/FID.err
#PBS -m abe
#PBS -M chtsha042@myuct.ac.za
cd /home/schetty1/
module purge 
ssh dtn
module load gcc/9.2.0
module load chpc/python/anaconda/3-2021.11
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/schetty1/.conda/envs/test_env/lib/
conda activate /home/schetty1/.conda/envs/test_env
python3 /home/schetty1/MedFIDAttempt2/FIDTests.py
conda deactivate
