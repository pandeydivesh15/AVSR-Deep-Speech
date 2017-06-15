import argparse
import glob
import sys
import os
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.exported_model import DeepSpeechModel

# Argument parser. This script expects 4 optional args.
parser = argparse.ArgumentParser(
	description='Run a trained Speech to Text model for some inputs. \
	**NOTE: Please use a trailing slash for directory names.')

parser.add_argument('-d', '--export_dir', type=str,
					help="Dir where the trained model's meta graph and data were exported")
parser.add_argument('-wd', '--wav_dir', type=str,
					help="Dir where wav files are stored (all files' transcripts will be generated)")
parser.add_argument('-f', '--wav_file', type=str,
					help="Wav file's location: Only one transcript generated. \
					If --wav_dir is given, --wav_file will have no effect.")
parser.add_argument('-n', '--model_name', type=str,
					help='Name of the model exported')

args = parser.parse_args()

export_dir = args.export_dir or 'data/export/00000001/'
model_name = args.model_name or 'export'

# Create DeepSpeechModel Class object. 
# This object will be responsible for handling tensorflow model.
model = DeepSpeechModel(export_dir, model_name)

model.restore_model()

# `wav_file_paths` is a list of strings. Each string signifies a file path for a .wav file.
if args.wav_dir:
	wav_file_paths = sorted(glob.glob(args.wav_dir + '*.wav'))
elif args.wav_file:
	wav_file_paths = [args.wav_file, ]
else:
	wav_file_paths = ['data/ldc93s1/LDC93S1.wav'] # default .wav file

transcripts = model.find_transcripts(wav_file_paths)

# `transcript` is a list of strings. 
# Each string is the transcript for the corresponding .wav file in `wav_file_paths`.
print transcripts

# Close tensorflow session
model.close()
