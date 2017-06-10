#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

python -u DeepSpeech_RHL.py \
  --train_files data/clean_data/train/data.csv \
  --dev_files data/clean_data/train/data.csv \
  --test_files data/clean_data/train/data.csv \
  --train_batch_size 40 \
  --dev_batch_size 10 \
  --test_batch_size 10 \
  --n_hidden 494 \
  --epoch 50 \
  --checkpoint_dir "$checkpoint_dir" \
  --export_dir data/export_model/
  "$@"
