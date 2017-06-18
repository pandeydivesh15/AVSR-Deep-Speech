import tensorflow as tf
import numpy as np

from util.text import ndarray_to_text
from util.audio import audiofile_to_input_vector
from util.spell import correction

# These constants must be same as those used during training.
# In main `DeepSpeech_RHL.py` script, 
# `NUM_MFCC_COEFF` is denoted by `n_input` and `N_CONTEXT` is denoted by `n_context`
NUM_MFCC_COEFF = 26
N_CONTEXT = 9


class DeepSpeechModel(object):
	"""Handles trained Deep Speech model"""
	def __init__(self, export_dir, model_name, use_spell_check=False):
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

	def restore_model(self):
		# Load meta graph and learned weights
		saver = tf.train.import_meta_graph(self.export_dir + self.name + '.meta')
		saver.restore(self.session, tf.train.latest_checkpoint(self.export_dir))

		# Get input and output nodes
		graph = tf.get_default_graph()
		self.input = graph.get_tensor_by_name("input_node:0")
		self.input_len = graph.get_tensor_by_name("input_lengths:0")
		self.output = graph.get_tensor_by_name("output_node:0")

	def find_transcripts(self, wav_file_paths):
		'''
		Args: 
			wav_file_paths:	A list containing filepaths for each wav file.
							Type of each element = str 
		'''
		transcripts = []

		# TODO: Currently, session.run() runs multiple times(once for each wav file). 
		# This is due to different batch sizes for each file.
		# Find a way to run the model only once.
		# Make batch size equal for all, and predict for entire batch at once.

		for path in wav_file_paths:
			source = np.array([(audiofile_to_input_vector(path, NUM_MFCC_COEFF , N_CONTEXT))])
			source_len = np.array([(len(source[-1]))])

			feed_dict = {self.input:source, self.input_len:source_len}

			batch_decoded = self.session.run(self.output, feed_dict)
			for decoded in batch_decoded[0]:
				if self.use_spell_check:
					transcripts.append(correction(ndarray_to_text(decoded)))
				else:
					transcripts.append(ndarray_to_text(decoded))

		return transcripts

	def close(self):
		# Close tensorflow session
		self.session.close()


