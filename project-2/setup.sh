#!/bin/bash
pip3 install -r requirements.txt

python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords');"

base_url='https://polybox.ethz.ch/index.php/s/UqbUKdkIQBhWzvI/download?path=%2F&files='
files=(	\
	'doc2vec_preprocessed.obj' \
	'sentiment_preprocessed.obj' \
	'embeddings-dim100.word2vec' \
	'mini.h5' \
	'word_embeddings.doc2vec' \
	'preprocessed.obj' \
	'preprocessed_w2v.obj' \
	'train_stories.csv' \
	'validation_stories.csv');

mkdir data
cd data

cmd="curl"
if [ -x "$(command -v curl)" ]; then
	echo 'Using cURL.\n'
	for f in "${files[@]}";
	do 
		echo "Downloading " $f "...\n"
		curl -o $f "$base_url$f"
	done
elif [ -x "$(command -v wget)" ]; then
	echo 'Using wget.\n'
	for f in "${files[@]}";
	do 
		echo "Downloading " $f "...\n"
		wget -O $f "$base_url$f"
	done
else
	echo 'Error: wget is not installed either. Please download the files manually from https://polybox.ethz.ch/index.php/s/UqbUKdkIQBhWzvI\n'
  	exit 1
fi

echo "-- Finished downloads. --"
