#!/bin/bash
NOW=$(date '+%F-%H-%M-%S')
srun \
  --job-name="cloudsen12" \
  --gpus=0 \
  --cpus-per-task=4 \
  --mem=150G \
  -p RTXA6000 \
  --container-mounts=/ds-sds/images/AI4EO/multi/cloudsen12/CloudSEN12-high:/data,/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset:/xarray/,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.03-py3.sqsh \
  --container-workdir="`pwd`" \
  --output="/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/src/couts/min_max.log" \
  --time=01:00:00 \
  --task-prolog="`pwd`/install.sh" \
  python global_minmax.py

##  --output="/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/couts/${NOW}_test.log" \