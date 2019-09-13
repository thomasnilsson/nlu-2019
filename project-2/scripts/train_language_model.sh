#!/bin/sh
cd ../src

python train.py --model_type lm --lr 2e-4 --epochs 15 --cell_size 1024 --cell_height 1 --batch_size 64
