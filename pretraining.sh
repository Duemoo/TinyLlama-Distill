#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -m lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --devices=2 \
    --num-nodes=1 \
    /scratch/e1506a02/TinyLlama-Distill/pretrain/tinyllama.py --devices 2 --train_data_dir /scratch/e1506a02/data/slim_processed_0.01 --val_data_dir  /scratch/e1506a02/data/slim_processed_0.01