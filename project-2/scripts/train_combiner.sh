#!/bin/sh
cd ../src

load_rnn="../saved_models/rnn/rnn_190607_163838/"
load_sentiment="../saved_models/sentiment/sentiment_190607_163344"
# load_lm="../saved_models/lm/lm_for_combiner"
load_doc2vec="../saved_models/doc2vec/doc2vec_190607_163517/"

# Best known parameters for training the combine
python3 train.py --model_type combiner --load_dirs $load_rnn $load_sentiment $load_doc2vec --epochs 10 --lr 1e-3 --hidden_size 128 #--fine_tune
