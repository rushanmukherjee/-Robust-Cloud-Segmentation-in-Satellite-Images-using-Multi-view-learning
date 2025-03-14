#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8 
#SBATCH --time=10:00:00
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_24.03-py3.sqsh
#SBATCH --container-mounts=/ds-sds/images/AI4EO/multi/cloudsen12/CloudSEN12-high:/data,/ds:/ds:ro,/home/rmukherjee:/home/rmukherjee
#SBATCH --container-workdir="/home/rmukherjee" 
#SBATCH --output=/home/rmukherjee/jupyter_logs/jupyter.log

python -m pip install -r /netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/requirements.txt

cat /etc/hosts
ifconfig
jupyter lab --ip=0.0.0.0 --port=8888
 