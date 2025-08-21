#!/bin/bash
#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=decoding_images
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=50:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=main

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

# Create a lookup table with the desired values
subjects=(9 10 11 12 13 14 15 16 17 18 19 20 21 22 23) #15 subs

# Extract combination for current task ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))
subject=${subjects[$task_id]}

echo "Running subject=$subject"

./default_decoding_images.sh $subject > default_decoding_images_sub_${subject}.txt 2>&1



