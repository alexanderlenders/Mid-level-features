#!/bin/bash
# ==============================================================================
# SLURM job script to perform encoding with JOBLIB (Single Node)
# ==============================================================================

#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=train_kinetics
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=2 # 1 process per GPU                                  
#SBATCH --cpus-per-task=16 # 16 CPU cores per process
#SBATCH --mem=60000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=90:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:2

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

num_workers=${SLURM_CPUS_PER_TASK}
num_gpus=2

./train_weight_decay.sh $num_workers $num_gpus > train_weight_decay.txt 2>&1

