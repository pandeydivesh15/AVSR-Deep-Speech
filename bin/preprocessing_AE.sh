#!/bin/sh
set -xe
if [ ! -f DeepSpeech_RHL.py ]; then
    echo "Please make sure you run this from main project's directory."
    exit 1
fi;

if [ ! -d data/AE_clean_data/ ]; then
    mkdir data/AE_clean_data
fi;

python ./bin/preprocess_auto_enc.py \
	./data/RHL_mp4/ \
	./data/AE_clean_data/ \
	--max_videos 2	\
	--screen_display
