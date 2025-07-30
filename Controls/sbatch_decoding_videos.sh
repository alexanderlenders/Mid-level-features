#!/bin/bash
#SBATCH --mail-user=alexandel91@zedat.fu-berlin.de   
#SBATCH --job-name=default_miniclips
#SBATCH --mail-type=ALL                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000 # specifies the maximum amount of memory in MB per node!                           
#SBATCH --time=01:00:00 # maximum time                           
#SBATCH --qos=standard
#SBATCH --partition=main

cd ./

source /home/alexandel91/.bashrc
conda activate encoding

# Create a lookup table with the desired values
subjects=(6 7 8 9 10 11 17 18 20 21 23 25 27 28 29 30 31 32 34 36)

# Extract combination for current task ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))
subject=${subjects[$task_id]}

echo "Running subject=$subject"

./default_decoding_videos.sh $subject > default_decoding_videos_sub_${subject}.txt 2>&1

