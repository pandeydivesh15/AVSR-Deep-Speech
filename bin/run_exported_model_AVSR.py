import argparse
import sys
import os
import subprocess
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from util.data_preprocessing_video import extract_and_store_visual_features
from util.exported_model import DeepSpeechModel
from util.video_stream import VideoStream 			


# Argument parser. This script expects 4 optional args.
parser = argparse.ArgumentParser(
	description='Run a trained Speech to Text model for some inputs. \
	**NOTE: Please use a trailing slash for directory names.')

parser.add_argument('-d', '--export_dir', type=str,
					help="Dir where the trained model's meta graph and data were exported")
parser.add_argument('-vf', '--video_file', type=str,
					help="Video file's location.")
parser.add_argument('-n', '--model_name', type=str,
					help='Name of the model exported')

parser.add_argument('--use_spell_check', default = False, action='store_true',
					help='Whether to use spell check system for decoded transcript from RNN')

args = parser.parse_args()

export_dir = args.export_dir or 'data/export_AVSR/00000001/'
model_name = args.model_name or 'export'

# Create DeepSpeechModel Class object. 
# This object will be responsible for handling tensorflow model.

# For video file, we will create a temporary audio file (.wav file) and json file (storing visual features)
temp_dir = '/tmp/'
temp_wav_name = 'temp.wav'
temp_json_name = 'temp.json'

# Using samplerate and bitrate similar to Red Hen Lab's sample video.
cmd = "ffmpeg -i " + args.video_file + " -ab 96k -ar 44100 -vn " + temp_dir + temp_wav_name
subprocess.call(cmd, shell=True)

# Now extract visual features from the video and store them in a JSON file.
extract_and_store_visual_features(args.video_file, temp_dir, temp_json_name)

model = DeepSpeechModel(export_dir, model_name, args.use_spell_check, use_visual_features=True)
model.restore_model()

transcript = model.find_transcripts(temp_dir+temp_wav_name, temp_dir+temp_json_name)

print "\n[Generated Transcript]\t", 
print transcript

# Delete temporary audio file and json file
os.remove(temp_dir+temp_wav_name)
os.remove(temp_dir+temp_json_name)

# Close tensorflow session
model.close()
