#!/bin/bash
#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=c12
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=00:20:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=main

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

./c12_differences_cnn.sh > c12_differences_cnn.txt 2>&1
