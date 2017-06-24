#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/caseHPC"))')
fi

python -u DeepSpeech_RHL.py \
  --train_files data/clean_data/train/data.csv \
  --dev_files data/clean_data/dev/data.csv \
  --test_files data/clean_data/test/data.csv \
  --train_batch_size 20 \
  --dev_batch_size 10 \
  --test_batch_size 10 \
  --n_hidden 494 \
  --epoch 50 \
  --dropout_rate 0.10 \
  --validation_step 5 \
  --display_step 5 \
  --checkpoint_dir "$checkpoint_dir" \
  --export_dir data/export/
  "$@"
