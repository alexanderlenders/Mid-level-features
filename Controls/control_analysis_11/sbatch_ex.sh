#!/bin/bash
# ==============================================================================
# SLURM job script to perform encoding with JOBLIB (Single Node)
# ==============================================================================

#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=feat_extraction
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=01:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=agcichy
#SBATCH --gres=gpu:1

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

./ex_c11.sh > ex_c11.txt 2>&1


