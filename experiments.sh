#!/bin/bash

GPU_ID=$1
DATASET=$2
NET=$3
TRAINorTESTorDEMO=$1

cd ~/Desktop/pytorch-faster-rcnn

source activate tensroflow-gpu

if [TRAINorTESTorDEMO == Train]; then
  ./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} ${DATASET} ${NET}
fi

if [TRAINorTESTorDEMO == TEST]; then
  ./experiments/scripts/test_faster_rcnn.sh ${GPU_ID} ${DATASET} ${NET}
fi

if [TRAINorTESTorDEMO == DEMO] then
  CUDA_VISIBLE_DEVICES=0 ./tools/demo.py \
    --net ${NET} \
    --dataset ${DATASET}
fi

