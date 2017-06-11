import argparse
import glob
import sys
import os
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.exported_model import DeepSpeechModel

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

model = DeepSpeechModel(export_dir, model_name)

model.restore_model()

if args.wav_dir:
	wav_file_paths = sorted(glob.glob(args.wav_dir + '*.wav'))
elif args.wav_file:
	wav_file_paths = [args.wav_file, ]
else:
	wav_file_paths = ['data/ldc93s1/LDC93S1.wav']

transcripts = model.find_transcripts(wav_file_paths)
print transcripts

# Close session
model.close()
