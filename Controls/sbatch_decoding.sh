#!/bin/bash
#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=decoding_images
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=04:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=main

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

./default_after_decoding.sh > default_after_decoding.txt 2>&1



