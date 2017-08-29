import argparse
import sys
import os
import subprocess
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.exported_model import DeepSpeechModel

# Argument parser. This script expects 4 optional args.
parser = argparse.ArgumentParser(
	description='Run a trained Speech to Text model for some inputs. \
	**NOTE: Please use a trailing slash for directory names.')

parser.add_argument('-d', '--export_dir', type=str,
					help="Dir where the trained model's meta graph and data were exported")
parser.add_argument('-af', '--wav_file', type=str,
					help="Wav file's location.")
parser.add_argument('-vf', '--video_file', type=str,
					help="Video file's location. \
					If --wav_file is also given as arg, --video_file will have no effect.")
parser.add_argument('-n', '--model_name', type=str,
					help='Name of the model exported')
parser.add_argument('--use_spell_check', default = False, action='store_true',
					help='Whether to use spell check system for decoded transcript from RNN')

args = parser.parse_args()

export_dir = args.export_dir or 'data/export/00000001/'
model_name = args.model_name or 'export'

# Create DeepSpeechModel Class object. 
# This object will be responsible for handling tensorflow model.
model = DeepSpeechModel(export_dir, model_name, args.use_spell_check)

model.restore_model()

if args.wav_file:
	wav_file_path = args.wav_file
elif args.video_file:
	# For video file, we will create a temporary audio file (.wav file)
	temp_file_path = '/tmp/temp.wav'
	# Using samplerate and bitrate similar to Red Hen Lab's sample video.
	cmd = "ffmpeg -i " + args.video_file + " -ab 96k -ar 44100 -vn " + temp_file_path
	subprocess.call(cmd, shell=True)

	wav_file_path = temp_file_path
else:
	wav_file_path = 'data/ldc93s1/LDC93S1.wav' # default .wav file

transcript = model.find_transcripts(wav_file_path)

print "\n[Generated Transcript]\t", 
print transcript

# Delete temporary audio file, if --video_file was given
if args.video_file:
	os.remove(temp_file_path)

# Close tensorflow session
model.close()
