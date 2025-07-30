#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --qos=standard
#SBATCH --partition=main
#SBATCH --time=02:30:00

# Get tunneling information
XDG_RUNTIME_DIR=""
node=$(hostname -s)
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Tunneling instructions
echo -e "
On your local computer, create an SSH tunnel:

  ssh -N -f -L ${port}:${node}:${port} ${USER}@curta.zedat.fu-berlin.de

On your local computer, point a browser to:

  http://localhost:${port}

and enter the password you created earlier.
"
source /home/alexandel91/.bashrc
conda activate encoding

tensorboard --logdir=/scratch/alexandel91/mid_level_features/results/CNN/training/ResNet18/resnet18_kinetics/ --port=${port} --host=${node}

