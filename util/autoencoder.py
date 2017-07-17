import tensorflow as tf
import numpy as np
import time 

from tqdm import tqdm

from util.RBM.util import tf_xavier_init

class AutoEncoder:
	def __init__(self, input_size, encoding_layer_sizes, layer_names):
		
		assert len(encoding_layer_sizes) == len(layer_names)

		self.layer_names = layer_names
		self.encoding_layer_sizes = encoding_layer_sizes

		self.x = tf.placeholder('float', [None, input_size], name = 'input')
		next_layer_input = self.x

		self.encoding_weights = []
		self.encoding_biases = []

		for i in range(len(self.encoding_layer_sizes)):
			input_dim = int(next_layer_input.get_shape()[1])
			layer_dim = self.encoding_layer_sizes[i]

			# Initialize W using xavier initialization
			W = tf.Variable(tf_xavier_init(input_dim, layer_dim, const=4.0), name=layer_names[i][0])

			# Initialize b to zero
			b = tf.Variable(tf.zeros([layer_dim]), name=layer_names[i][1])

			self.encoding_weights.append(W)
			self.encoding_biases.append(b)

			# Output of this layer becomes next layer's input
			next_layer_input = tf.nn.sigmoid(tf.matmul(next_layer_input, W) + b)

		self.encoded_x = next_layer_input
		
		self.encoding_layer_sizes.reverse()
		self.decoding_layer_sizes = self.encoding_layer_sizes[1:] + [input_size]
		self.encoding_layer_sizes.reverse()

		reversed_encoding_weights = self.encoding_weights[::-1] # Reversed list
		self.decoding_weights = []
		self.decoding_biases = []

		for i in range(len(self.decoding_layer_sizes)):
			input_dim = int(next_layer_input.get_shape()[1])
			layer_dim = self.decoding_layer_sizes[i]

			# Take transpose of previous encoding layer's weights.
			W = tf.identity(tf.transpose(reversed_encoding_weights[i]))

			b = tf.Variable(tf.zeros([layer_dim]))

			self.decoding_weights.append(W)
			self.decoding_biases.append(b)

			next_layer_input = tf.nn.sigmoid(tf.matmul(next_layer_input, W) + b)

		# the fully reconstructed value of x is here:
		self.reconstructed_x = next_layer_input

		self.decoding_weights.reverse() 
		# Reversing helps us while restoring RBM weights. See `self.load_rbm_weights()` func.

		# Map layer names to actual layer weights and biases. Construct dict.
		self.map_layer_with_names()

		# Computing Cost
		self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
		self.optimizer = self.get_optimizer().minimize(self.cost)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def get_optimizer(self):
		return tf.train.AdamOptimizer()

	def transform(self, batch_x):
		return self.sess.run(self.encoded_x, feed_dict={self.x: batch_x})

	def reconstruct(self, batch_x):
		return self.sess.run(self.reconstructed_x, feed_dict={self.x: batch_x})

	def get_cost(self, batch_x):
		return self.sess.run(self.cost, feed_dict={self.x: batch_x})

	def partial_fit(self, batch_x):
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: batch_x})
		return cost

	def fit(self, 
			data_x_train,
			data_x_dev=None,
			data_x_test=None,
			n_epochs=10,
			batch_size=10):
		assert n_epochs > 0
		assert batch_size < data_x_train.shape[0]

		size_x_train = data_x_train.shape[0]

		n_batches = size_x_train / batch_size
		
		for e in range(n_epochs):
			epoch_costs = np.zeros(n_batches)
			bar = tqdm(range(n_batches), desc='Epoch: {:d}'.format(e))

			for i in bar:
				batch_x = data_x_train[i*batch_size:(i+1)*batch_size]
				err = self.partial_fit(batch_x)
				epoch_costs[i] = err

			mean_cost = epoch_costs.mean()
			print 'Train error: {:.4f}'.format(mean_cost)

			if data_x_dev is not None:
				random_indices = np.random.randint(0, data_x_dev.shape[0], batch_size)
				batch_x = data_x_dev[random_indices]
				err = self.partial_fit(batch_x)
				print 'Validation data error: {:.4f}'.format(err)

		if data_x_test is not None:
				err = self.partial_fit(data_x_test)
				print 'Test data error: {:.4f}'.format(err)

	def load_rbm_weights(self, path, layer_names):
		assert layer_names == self.layer_names

		data_dict = {}
		for i in len(self.encoding_layer_sizes):
			 data_dict[self.layer_names[i][0]] = self.encoding_weights[i]
			 data_dict[self.layer_names[i][1]] = self.encoding_biases[i]

		saver = tf.train.Saver(data_dict)
		saver.restore(self.sess, path)

		# Now, we must also load decoding weights.
		for i in len(self.decoding_layer_sizes):
			self.sess.run(self.decoding_weights[i].assign(tf.transpose(self.encoding_weights[i]))) 

	def map_layer_with_names(self):
		self.layer_map = {}

		for i in range(len(self.encoding_layer_sizes)):
			# Encoding layers
			self.layer_map[self.layer_names[i][0]] = self.encoding_weights[i]
			self.layer_map[self.layer_names[i][1]] = self.encoding_biases[i]
			# Decoding layers
			self.layer_map[self.layer_names[i][0] + 'd'] = self.decoding_weights[i]
			self.layer_map[self.layer_names[i][1] + 'd'] = self.decoding_biases[i]

	def load_parameters(self, path):
		saver = tf.train.Saver(self.layer_map)
		saver.restore(self.sess, path)

	def save_parameters(self, path):
		saver = tf.train.Saver(self.layer_map)
		saver.save(self.sess, path)
