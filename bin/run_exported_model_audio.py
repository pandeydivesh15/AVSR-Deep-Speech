import argparse
import glob
import sys
import os
import subprocess
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.exported_model_audio import DeepSpeechModel

# Argument parser. This script expects 4 optional args.
parser = argparse.ArgumentParser(
	description='Run a trained Speech to Text model for some inputs. \
	**NOTE: Please use a trailing slash for directory names.')

parser.add_argument('-d', '--export_dir', type=str,
					help="Dir where the trained model's meta graph and data were exported")
parser.add_argument('-wd', '--wav_dir', type=str,
					help="Dir where wav files are stored (all files' transcripts will be generated)")
parser.add_argument('-af', '--wav_file', type=str,
					help="Wav file's location. Only one transcript generated. \
					If --wav_dir is given, --wav_file will have no effect.")
parser.add_argument('-vf', '--video_file', type=str,
					help="Video file's location. Only one transcript generated. \
					If --wav_dir or --wav_file are also given as args, --video_file will have no effect.")
parser.add_argument('-n', '--model_name', type=str,
					help='Name of the model exported')
parser.add_argument('--use_spell_check', default = False, action='store_true',
					help='Whether to use spell check system for decoded transcripts from RNN')

args = parser.parse_args()

export_dir = args.export_dir or 'data/export/00000001/'
model_name = args.model_name or 'export'

# Create DeepSpeechModel Class object. 
# This object will be responsible for handling tensorflow model.
model = DeepSpeechModel(export_dir, model_name, args.use_spell_check)

model.restore_model()

# `wav_file_paths` is a list of strings. Each string signifies a file path for a .wav file.
if args.wav_dir:
	wav_file_paths = sorted(glob.glob(args.wav_dir + '*.wav'))
elif args.wav_file:
	wav_file_paths = [args.wav_file, ]
elif args.video_file:
	# For video file, we will create a temporary audio file (.wav file)
	temp_file_path = '/tmp/temp.wav'
	# Using samplerate and bitrate similar to Red Hen Lab's sample video.
	cmd = "ffmpeg -i " + args.video_file + " -ab 96k -ar 44100 -vn " + temp_file_path
	subprocess.call(cmd, shell=True)

	wav_file_paths = [temp_file_path, ]
else:
	wav_file_paths = ['data/ldc93s1/LDC93S1.wav'] # default .wav file

transcripts = model.find_transcripts(wav_file_paths)

# `transcript` is a list of strings. 
# Each string is the transcript for the corresponding .wav file in `wav_file_paths`.
print transcripts

# Delete temporary audio file, if --video_file was given
if args.video_file:
	os.remove(temp_file_path)

# Close tensorflow session
model.close()
