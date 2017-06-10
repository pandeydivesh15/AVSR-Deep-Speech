#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from main project's directory."
    exit 1
fi;

if [ ! -d data/clean_data/ ]; then
    mkdir data/clean_data
    mkdir data/clean_data/train data/clean_data/dev data/clean_data/test 
fi;

if [ ! -d data/clean_data/train ]; then
    mkdir data/clean_data/train
fi;

if [ ! -d data/clean_data/dev ]; then
    mkdir data/clean_data/dev
fi;

if [ ! -d data/clean_data/test ]; then
    mkdir data/clean_data/test
fi;

python ./bin/preprocess_data.py \
	data/clean_data/train/ \
	data/clean_data/dev/ \
	data/clean_data/test/ \
	0.90 \
	0.05 \
	0.05