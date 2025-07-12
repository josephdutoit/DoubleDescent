#!/bin/bash

MAX_WIDTH=40
NUM_GPUS=1
NUM_WORKERS=7
CONFIG="lin_reg_config"

for i in $(seq 1 $MAX_WIDTH)
do
    GPU_ID=$(((i - 1) % NUM_GPUS))

    echo "Launching run $i on GPU $GPU_ID"

    python3 train.py --gpu_id $GPU_ID --train_samples $i --num_workers $NUM_WORKERS --config $CONFIG &

    if [[ $(($i % $NUM_GPUS)) -eq 0 ]]; then
        echo "Waiting for batch of jobs to finish..."
        wait
        echo "Batch finished."
    fi
done
