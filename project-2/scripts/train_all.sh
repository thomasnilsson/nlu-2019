#!/bin/sh

# Go to src dir
cd ../src

echo "Training sentiment model"
python3 train.py --model_type sentiment --epochs 3 --save_name sentiment_for_combiner

echo "Training rnn model"
python3 train.py --model_type rnn --lr 5e-5 --epochs 5 --cell_size 160 --hidden_size 128 --dropout 0.3 --save_name rnn_for_combiner

echo "Training lm model"
python3 train.py --model_type lm --lr 2e-4 --epochs 10 --cell_size 1024 --cell_height 1 --batch_size 64 --save_name lm_for_combiner

echo "Training doc2vec"
python3 train.py --model_type doc2vec --save_name doc2vec_for_combiner

# Paths to saved models
load_rnn="../saved_models/rnn/rnn_for_combiner"
load_sentiment="../saved_models/sentiment/sentiment_for_combiner"
load_lm="../saved_models/lm/lm_for_combiner"
load_doc2vec="../saved_models/doc2vec/doc2vec_for_combiner/"

echo "Training combiner"
python3 train.py --model_type combiner --load_dirs $load_rnn  $load_sentiment $load_lm $load_doc2vec --epochs 10 --lr 1e-3 --hidden_size 128 --save_name complete #--fine_tune ?
# $load_lm
echo "Testing combined model"
load_combiner="../saved_models/combiner/complete"
python3 test.py --model_type combiner --load $load_combiner
