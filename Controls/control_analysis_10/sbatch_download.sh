#!/bin/bash
#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=download_kinetics
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                  
#SBATCH --cpus-per-task=2
#SBATCH --mem=50000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=72:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=main

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

num_workers=${SLURM_CPUS_PER_TASK}

./download_kinetics_train.sh $num_workers > download_kinetics_train.txt 2>&1

