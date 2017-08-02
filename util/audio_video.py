import json

import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

def get_audio_visual_feature_vector(audio_file_path, json_file_path, numcep, numcontext):
	with open(json_file_path, 'r') as f:
		data = json.loads(f.read())
		file_name = data['name']
		visual_feature = np.array(data['encoded'])

	# `visual_feature` is a numpy array which stores visual bottleneck features for each split .wav audio file.
	# Shape of `visual_feature` = (x, y)
	# Here, x = number of video frames to be used for training.
	# And y = size of encodings(for each frame) extracted from our deep autoencoder.
	# In our case y = 50
	# Please see `util.data_preprocessing_video.py` for more details.

	# Load wav file
	fs, audio = wav.read(audio_file_path)

	# Get mfcc coefficients
	orig_inputs = mfcc(audio, samplerate=fs, numcep=26)

	# We only keep every 8th feature (BiRNN stride = 8)
	orig_inputs = orig_inputs[::8]
	# This is done so that we can match with FPS of our video.
	# For our video, normal frame rate is 25, or time difference between each frame is 0.04s(approx).
	# The above function returns MFCC features at every 0.005 time step.
	# For having equal audio and visual features, we must extract MFCC features after every 0.04 secs.
	
	assert len(orig_inputs) == visual_feature.shape[0]
		
	orig_inputs = (orig_inputs - np.mean(orig_inputs))/np.std(orig_inputs)
	visual_feature = (visual_feature - np.mean(visual_feature))/np.std(visual_feature)
	modified_inputs = np.hstack((orig_inputs, visual_feature))

	# The next section of code is mostly similar to the one in `util.audio.audiofile_to_input_vector()` func.

	# For each time slice of the training set, we need to copy the context this makes
	# the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
	# because of:
	#  - numcep dimensions for the current mfcc feature set
	#  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
	# => so numcep + 2*numcontext*numcep
	train_inputs = np.array([], np.float32)
	train_inputs.resize((modified_inputs.shape[0], numcep + 2*numcep*numcontext))

	empty_feature = np.array([])
	empty_feature.resize((numcep))

	# Prepare train_inputs with past and future contexts
	time_slices = list(range(train_inputs.shape[0]))
	context_past_min   = time_slices[0]  + numcontext
	context_future_max = time_slices[-1] - numcontext
	for time_slice in time_slices:
		### Reminder: array[start:stop:step]
		### slices from indice |start| up to |stop| (not included), every |step|
		# Pick up to numcontext time slices in the past, and complete with empty visual features
		need_empty_past     = max(0, (context_past_min - time_slice))
		empty_source_past   = list(empty_feature for empty_slots in range(need_empty_past))
		data_source_past    = modified_inputs[max(0, time_slice - numcontext):time_slice]
		assert(len(empty_source_past) + len(data_source_past) == numcontext)

		# Pick up to numcontext time slices in the future, and complete with empty visual features
		need_empty_future   = max(0, (time_slice - context_future_max))
		empty_source_future = list(empty_feature for empty_slots in range(need_empty_future))
		data_source_future  = modified_inputs[time_slice + 1:time_slice + numcontext + 1]
		assert(len(empty_source_future) + len(data_source_future) == numcontext)

		if need_empty_past:
			past   = np.concatenate((empty_source_past, data_source_past))
		else:
			past   = data_source_past

		if need_empty_future:
			future = np.concatenate((data_source_future, empty_source_future))
		else:
			future = data_source_future

		past   = np.reshape(past, numcontext*numcep)
		now    = modified_inputs[time_slice]
		future = np.reshape(future, numcontext*numcep)

		train_inputs[time_slice] = np.concatenate((past, now, future))
		assert(len(train_inputs[time_slice]) == numcep + 2*numcep*numcontext)

	# Return final train inputs
	return train_inputs

