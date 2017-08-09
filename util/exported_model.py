import tensorflow as tf
import numpy as np

from util.text_RHL import ndarray_to_text
from util.audio import audiofile_to_input_vector
from util.audio_video import get_audio_visual_feature_vector
from util.spell import correction
from util.autoencoder import AutoEncoder

# These constants must be same as those used during training.\
# FOR AUDIO-ONLY SPEECH RECOG
# `NUM_MFCC_COEFF` must be equal to `n_input` in the main `DeepSpeech_RHL.py`. (Default = 26)

# FOR AUDIO-VISUAL SPEECH RECOG
# `NUM_VISUAL` must be equal to number of visual features used for each time step. (Default = 50)
# `NUM_MFCC_COEFF` + `NUM_VISUAL` = `n_input`, where `n_input` is from main `DeepSpeech_RHL_AVSR.py`.

# In main `DeepSpeech_RHL.py` or `DeepSpeech_RHL_AVSR.py`  script, `N_CONTEXT` is denoted by `n_context`

NUM_MFCC_COEFF = 26
N_CONTEXT = 9
NUM_VISUAL = 50

class DeepSpeechModel(object):
	"""Handles trained Deep Speech model(for audio-only / audio-visual  speech recognition)"""
	def __init__(self, export_dir, model_name, use_spell_check=False, use_visual_features=False):
		'''
		Args:
			export_dir(type = str):	Path to directory where trained model 
									has been exported (with trailing slash).
			model_name(type = str):	Name of the model exported.
		'''
		self.export_dir = export_dir
		self.session = tf.Session()
		self.name = model_name
		self.use_spell_check = use_spell_check

		self.use_visual_features = use_visual_features


	def restore_model(self):
		# Load meta graph and learned weights
		saver = tf.train.import_meta_graph(self.export_dir + self.name + '.meta')
		saver.restore(self.session, tf.train.latest_checkpoint(self.export_dir))

		# Get input and output nodes
		graph = tf.get_default_graph()
		self.input = graph.get_tensor_by_name("input_node:0")
		self.input_len = graph.get_tensor_by_name("input_lengths:0")
		self.output = graph.get_tensor_by_name("output_node:0")

	def find_transcripts(self, wav_file_path, visual_feature_json_path=None):
		'''
		Args: 
			wav_file_path:		the filepath for your wav file.
			visual_features:	Visual features for video based speech recognition.
								These will be required when the exported model is of AVSR type.
							 
		'''
		if self.use_visual_features:
			assert visual_feature_json_path is not None

			source = np.array([(get_audio_visual_feature_vector(
				wav_file_path, 
				visual_feature_json_path,
				NUM_MFCC_COEFF+NUM_VISUAL, 
				N_CONTEXT))])
		else:
			source = np.array([(audiofile_to_input_vector(wav_file_path, NUM_MFCC_COEFF , N_CONTEXT))])
		source_len = np.array([(len(source[-1]))])

		feed_dict = {self.input:source, self.input_len:source_len}

		decoded = self.session.run(self.output, feed_dict)[0][0]
		# session.run() will return shape = (1,1,X). X = Number of characters in the transcript

		transcript = ndarray_to_text(decoded)
		if self.use_spell_check:
			transcript = correction(transcript)

		return transcript

	def close(self):
		# Close tensorflow session
		self.session.close()


