#!/bin/bash

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 8 --train_data_dir /home/hoyeon/TinyLlama-Distill/data/slim_processed_0.01