#!/bin/bash

GPU_ID=$1
DATASET=$2
NET=$3
TRAINorTESTorDEMO=$4

cd ~/Desktop/pytorch-faster-rcnn


#conda activate tensroflow-gpu

if [ "${TRAINorTESTorDEMO}" == "TRAIN" ]; then
    ./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} ${DATASET} ${NET}
fi

if [ "${TRAINorTESTorDEMO}" == "TEST" ]; then
    ./experiments/scripts/test_faster_rcnn.sh ${GPU_ID} ${DATASET} ${NET}
fi

if [ "${TRAINorTESTorDEMO}" == "DEMO" ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py \
    --net ${NET} \
    --dataset ${DATASET}
fi

