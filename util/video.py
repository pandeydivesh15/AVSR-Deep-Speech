import json

import numpy as np

def get_visual_feature_vector(json_file_path, numcep=50, numcontext):
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

	# Implementation left

