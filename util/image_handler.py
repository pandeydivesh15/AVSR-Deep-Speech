import random
import glob

import numpy as np
import cv2
from scipy.ndimage import imread

IMAGE_WIDTH = IMAGE_HEIGHT = 32

def visualize_image(image, name="Image", resize=False, save_image=False, path=None):
	image = image.reshape([IMAGE_WIDTH, IMAGE_HEIGHT])
	image = image.astype(np.uint8)

	if resize: 
		image = cv2.resize(image, (IMAGE_WIDTH * 10, IMAGE_HEIGHT * 10))

	cv2.imshow(name, image)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()

	if save_image:
		assert path is not None
		cv2.imwrite(path, image)


class ImageDataSet():
	"""
	Class for handling and preparing images before training any neural network.
	In this project, we will be working with mouth region images.
	"""

	def __init__(self, image_dir):
		self.image_dir = image_dir

		self.train = None
		self.dev = None
		self.test = None

	def read_data(self, train_split=0.80, dev_split=0.10, test_split=0.10):
		assert (train_split + dev_split + test_split == 1.0)

		all_images = glob.glob(self.image_dir + "*.png")
		data = []

		for image_path in all_images:
			image = imread(image_path, flatten=True)
			image = image.reshape(IMAGE_WIDTH*IMAGE_HEIGHT)
			# image = np.multiply(image, 1.0 / 255.0) No scaling here

			data.append(image)

		data = np.array(data)
		data = data.astype(np.uint8)

		total_images = data.shape[0]

		train_limit = int(total_images * train_split)
		dev_limit = train_limit + int(total_images * dev_split)
		
		self.train = data[:train_limit]
		self.dev = data[train_limit:dev_limit]
		self.test = data[dev_limit:]

		# Only shuffling training data.
		random.shuffle(self.train)

		self.data_dict = {
			'train':	self.train,
			'dev':		self.dev,
			'test':		self.test}

	def get_data(self, choice='train'):
		assert (choice == 'train' or choice == 'dev' or choice == 'test')
		assert self.train is not None

		return self.data_dict[choice]












