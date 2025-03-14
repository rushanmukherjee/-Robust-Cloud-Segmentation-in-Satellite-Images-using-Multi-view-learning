#!/bin/bash
NOW=$(date '+%F-%H-%M-%S')
srun \
  --job-name="cloudsen12" \
  --gpus=1 \
  --cpus-per-task=8 \
  --mem=100G \
  -p A100-40GB \
  --container-mounts=/ds-sds/images/AI4EO/multi/cloudsen12/CloudSEN12-high:/data,/netscratch/rmukherjee/mlflow_logs:/mlflow_logs,/netscratch/:/netscratch/,/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset:/xarray,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.03-py3.sqsh \
  --container-workdir="`pwd`" \
  --output="/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/src/couts/AMG_AD_eval.log" \
  --time=02:00:00 \
  --task-prolog="`pwd`/install.sh" \
  python eval_seg.py -c config25.yaml -r /netscratch/rmukherjee/mlflow_logs/754613372771536259/6ce5d66ec1c943898ce0c83ee4a1460a

##  --output="/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/couts/${NOW}_test.log" \