import os
import argparse
import sys
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import data_preprocessing_autoencoder

# Argument parser. This script expects 2 necessory positional args and 2 optional args.
parser = argparse.ArgumentParser(
			description='Preprocesses videos and extracts mouth regions for training Auto Encoder')

parser.add_argument('video_dir', type=str,
					help='Directory path where all videos are stored.')
parser.add_argument('output_dir', type=str,
					help='Output dir for storing processed images (with trailing slash)')


parser.add_argument('--max_videos', default = 1, type=int,
					help='Limit on number of videos to be used for preprocessing')
parser.add_argument('--screen_display', default = False, action='store_true',
					help='Determines whether to display the video being processed.')


args = parser.parse_args()

status = data_preprocessing_autoencoder.prepare_data(
			args.video_dir,
			args.output_dir,
			args.max_videos,
			args.screen_display)
if status:
	print "[INFO]: Images of mouth regions(32 X 32) extracted and stored at `args.output_dir`"
