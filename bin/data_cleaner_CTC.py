import sys
import os
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd

from util.audio_video import get_audio_visual_feature_vector
from util.text_RHL import text_to_char_array

# The following variables must have values same as the ones while training model. 
# See `DeepSpeech_RHL.py` / `DeepSpeech_RHL_AVSR.py` for more details.

N_CONTEXT = 9
NUMCEP = 50+26 # For AVSR, NUMCEP = 76 and for audio-only model, NUMCEP = 26

def clean_data_for_CTC(path):
	"""
	For applying CTC algorithm, we must ensure that for any training data, the target sequence
	is at most as long as the input sequence. In our case, an error can arise of the form (while training):

	ERROR: "Not enough time for target transition sequence (required: XX, available: YY). 
			You can turn this error into a warning by using the flag ignore_longer_outputs_than_inputs"

	This error is expected to come while training AVSR model. This is because in that case, 
	length of output transcript (output sequence) may be greater then the input feature vector.

	Size of input feature vector in the case of AVSR is small because we need to use both audio 
	and visual info in any time step. Due to small FPS (30), size of input feature vector may 
	be small in some cases. Please see `util.audio_video.get_audio_visual_feature_vector` for details.

	This function removes any data point which can create problems while training. 
	The function removes entry from main CSV file.

	"""

	data_frame = pd.read_csv(path)

	indices_to_delete = [] 
	# Stores the indices which must be removed from the main CSV file.

	for index, row in data_frame.iterrows():
		output_transcript = text_to_char_array(row['transcript'])

		audio_path = row['wav_filename']
		json_path = row['wav_filename'].split('.')[0] + '.json'
		feature = get_audio_visual_feature_vector(audio_path, json_path , NUMCEP, N_CONTEXT)

		if output_transcript.shape[0] > feature.shape[0]:
			indices_to_delete.append(index)

	data_frame.drop(data_frame.index[indices_to_delete], inplace=True)

	data_frame.to_csv(path, sep=",", index=None) # Save CSV

if __name__ == '__main__':
	data_dir = 'data/clean_data/'

	clean_data_for_CTC(data_dir + 'train/data.csv')
	clean_data_for_CTC(data_dir + 'dev/data.csv')
	clean_data_for_CTC(data_dir + 'test/data.csv')
	
