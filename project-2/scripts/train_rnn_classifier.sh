#!/bin/sh
cd ../src

# Best known parameters for training the rnn_classifier model
python train.py --model_type rnn --lr 5e-5 --epochs 10 --cell_size 160 --hidden_size 128 --dropout 0.3
