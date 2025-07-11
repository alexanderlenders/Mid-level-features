#!/bin/bash
# ==============================================================================
# SLURM job script to perform encoding with JOBLIB (Single Node)
# ==============================================================================

#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=download_kinetics
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                  
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=08:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:2

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

num_workers=${SLURM_CPUS_PER_TASK}
num_gpus=2

./sbatch_train.sh $num_workers $num_gpus > train.txt 2>&1

