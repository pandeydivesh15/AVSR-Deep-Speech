#!/bin/sh
set -xe
if [ ! -f DeepSpeech_RHL.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d data/AE_clean_data/ ]; then
    echo "Make sure you have all mouth region images stored at data/AE_clean_data/ directory."
fi;

if [ ! -d data/AE_and_RBM_model_saves/ ]; then
    mkdir data/AE_and_RBM_model_saves
fi;

python ./bin/AE_training.py \
	./data/AE_clean_data/ \
	./data/AE_and_RBM_model_saves/	\
	--visualize