import random
import glob

import numpy as np
from scipy.ndimage import imread
import cv2
import matplotlib.pyplot as plt

IMAGE_WIDTH = IMAGE_HEIGHT = 32

def visualize_image(image, rescale = True):
	if rescale: 
		image = np.multiply(image, 255.0)
	image = image.reshape([IMAGE_WIDTH, IMAGE_HEIGHT])
	
	# Implementation left


class ImageDataSet():
	"""
	Class for handling and preparing images before training any neural network.
	In this project, we will be working with mouth region images.
	"""

	def __init__(self, image_dir, normalize=True):
		self.image_dir = image_dir

		self.train = None
		self.dev = None
		self.test = None

		self.normalize = normalize 
		# If true: We scale data to have mean = 0, std_dev = 1
		# Standard normally distributed data: Gaussian with zero mean and unit variance.

	def read_data(self, train_split=0.80, dev_split=0.10, test_split=0.10, ):
		assert (train_split + dev_split + test_split == 1.0)

		all_images = glob.glob(self.image_dir + "*.png")
		data = []

		for image_path in all_images:
			image = imread(image_path, flatten=True)
			image = image.reshape(IMAGE_WIDTH*IMAGE_HEIGHT)
			image = np.multiply(image, 1.0 / 255.0)

			if self.normalize:
				mean = np.mean(image)
				std_dev = np.std(image)
				image = (image - mean) / std_dev

			data.append(image)

		random.shuffle(data)
		data = np.array(data)

		total_images = data.shape[0]

		train_limit = int(total_images * train_split)
		dev_limit = train_limit + int(total_images * dev_split)
		
		self.train = data[:train_limit]
		self.dev = data[train_limit:dev_limit]
		self.test = data[dev_limit:]

		self.data_dict = {
			'train':	self.train,
			'dev':		self.dev,
			'test':		self.test}

	def get_data(self, choice='train'):
		assert (choice == 'train' or choice == 'dev' or choice == 'test')
		assert self.train is not None

		return self.data_dict[choice]












