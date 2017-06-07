import os
import sys
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import data_preprocessing

audio_dir = "data/RHL_wav/"
json_dir = "data/RHL_json/"
output_dir = "data/clean_data/train/"

data_preprocessing.split_aligned_audio(audio_dir, json_dir, output_dir)
data_preprocessing.create_csv(output_dir)

