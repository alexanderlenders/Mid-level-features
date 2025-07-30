#!/bin/bash
# ==============================================================================
# SLURM job script to perform encoding with JOBLIB (Single Node)
# ==============================================================================

#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=c6_1
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=30000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=04:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=main

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

./run_c6_1_images.sh > c6_1_images.txt 2>&1


