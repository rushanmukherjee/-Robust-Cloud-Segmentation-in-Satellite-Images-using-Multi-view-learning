#!/bin/bash
NOW=$(date '+%F-%H-%M-%S')
srun \
  --job-name="cloudsen12" \
  --gpus=1 \
  --cpus-per-task=8 \
  --mem=100G \
  -p RTX3090 \
  --container-mounts=/ds-sds/images/AI4EO/multi/cloudsen12/CloudSEN12-high:/data,/netscratch/rmukherjee/mlflow_logs:/mlflow_logs,/netscratch/:/netscratch/,/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset:/xarray,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.03-py3.sqsh \
  --container-workdir="`pwd`" \
  --output="/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/src/couts/2MC_eval.log" \
  --time=02:00:00 \
  --task-prolog="`pwd`/install.sh" \
  python eval_seg2Modality.py -c config24.yaml -r /netscratch/rmukherjee/mlflow_logs/510770825602285381/85687e6af5fc466e943212c9779a548e

##  --output="/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/couts/${NOW}_test.log" \