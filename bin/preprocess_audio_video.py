import os
import argparse
import sys
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import data_preprocessing
from util import data_preprocessing_video

# Argument parser. This script expects 6 necessory positional args.
parser = argparse.ArgumentParser(
	description='Preprocesses video data for training audio and visual speech recognition model')

parser.add_argument('output_dir_train', type=str,
					help='Output dir for storing training files (with trailing slash)')
parser.add_argument('output_dir_dev', type=str,
					help='Output dir for storing files for validation (with trailing slash)')
parser.add_argument('output_dir_test', type=str,
					help='Output dir for storing test files (with a trailing slash)')
parser.add_argument('train_split', type=float,
					help='a float value for deciding percentage of data split for training the model')
parser.add_argument('dev_split', type=float,
					help='a float value for deciding percentage of validation data')
parser.add_argument('test_split', type=float,
					help='a float value for deciding percentage of test data')

args = parser.parse_args()

# Set default locations
video_dir = "data/RHL_mp4/"
audio_dir = "data/RHL_wav/"

# Percentages/ratios should add up to 1.0 
if args.train_split + args.dev_split + args.test_split != 1.0:
	print "Make sure that train, test, and dev split ratios add upto 1.0"
	exit()

# Convert any mp4 found in data/RHL_mp4/ and store it in data/RHL_wav/
data_preprocessing.convert_mp4(video_dir, audio_dir)
print "[INFO]: Main video files converted to .wav files."

# Generates data for audio-visual speech recognition
# Visual encodings are stored as JSON files and audio data as .wav files
data_preprocessing_video.preprocess_videos(
	args.output_dir_train,
	args.output_dir_dev,
	args.output_dir_test,
	args.train_split,
	args.dev_split,
	args.test_split)
print "[INFO]: Train/Dev/Test files generated. Now creating CSVs."

# Create CSVs out of all split files.
data_preprocessing.create_csv(args.output_dir_train)
data_preprocessing.create_csv(args.output_dir_dev)
data_preprocessing.create_csv(args.output_dir_test)
print "[INFO]: CSV files generated at train/ dev/ test/ locations. Exiting."
