import os
import argparse
import sys
import random
# Make sure that we can import functions/classes from utils/ folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from util.RBM import BBRBM, BGRBM, GBRBM
from util.autoencoder import AutoEncoder
from util.image_handler import ImageDataSet, visualize_image

def std_normalize(batch):
    norm_batch = np.zeros(batch.shape)
    for i in range(len(batch)):
        norm_batch[i] = (batch[i] - np.mean(batch[i])) / np.std(batch[i])
    return norm_batch

# Argument parser. This script expects 2 necessory positional args.
parser = argparse.ArgumentParser(
	description='Builds and Trains our autoencoder, as required in the paper' + 
	'https://ibug.doc.ic.ac.uk/media/uploads/documents/petridispantic_icassp2016.pdf')

parser.add_argument('data_dir', type=str,
					help='Directory storing all 32*32 mouth region images (dir with trailing slash(\))')
parser.add_argument('save_model_dir', type=str,
					help="Directory where RBM's parameters and autoencoder's parameters will be saved")
parser.add_argument('--visualize', default = False, action='store_true',
					help='If true, a random image from test data is shown along with its reconstruction.')

args = parser.parse_args()

dataset = ImageDataSet(args.data_dir)
dataset.read_data(train_split=0.80, dev_split=0.10, test_split=0.10)
images_train_norm = std_normalize(dataset.get_data('train'))
images_dev_norm = std_normalize(dataset.get_data('dev'))

images_test = dataset.get_data('test')
images_test_norm = std_normalize(dataset.get_data('test'))

n_input = images_train_norm.shape[1] # According to the paper, this must be equal to 32 * 32 = 1024
gbrbm_1 = GBRBM(n_input, 2000, learning_rate=0.001, use_tqdm=True, sigma=1)
errs = gbrbm_1.fit(images_train_norm, n_epoches=20, batch_size=100)

bbrbm_1 = BBRBM(2000, 1000, learning_rate=0.1, use_tqdm=True)
output = gbrbm_1.transform(images_train_norm)
errs = bbrbm_1.fit(output, n_epoches=20, batch_size=100)

bbrbm_2 = BBRBM(1000, 500, learning_rate=0.1, use_tqdm=True)
output = bbrbm_1.transform(output)
errs = bbrbm_2.fit(output, n_epoches=20, batch_size=100)

bgrbm_1 = BGRBM(500, 50, learning_rate=0.001, use_tqdm=True, sigma=1)
output = bbrbm_2.transform(output)
errs = bgrbm_1.fit(output, n_epoches=20, batch_size=100)

# Save all the learned parameters
gbrbm_1.save_weights(args.save_model_dir+'gbrbm_1', 'gbrbm_1')
bbrbm_1.save_weights(args.save_model_dir+'bbrbm_1', 'bbrbm_1')
bbrbm_2.save_weights(args.save_model_dir+'bbrbm_2', 'bbrbm_2')
bgrbm_1.save_weights(args.save_model_dir+'bgrbm_1', 'bgrbm_1')

auto_enc = AutoEncoder(
			n_input, 
			encoding_layer_sizes=[2000, 1000, 500, 50], 
			layer_names=[['gbrbm_1_w', 'gbrbm_1_h'],
						 ['bbrbm_1_w', 'bbrbm_1_h'],
						 ['bbrbm_2_w', 'bbrbm_2_h'],
						 ['bgrbm_1_w', 'bgrbm_1_h']])
auto_enc.load_rbm_weights(args.save_model_dir+'gbrbm_1', 0)
auto_enc.load_rbm_weights(args.save_model_dir+'bbrbm_1', 1)
auto_enc.load_rbm_weights(args.save_model_dir+'bbrbm_2', 2)
auto_enc.load_rbm_weights(args.save_model_dir+'bgrbm_1', 3)

auto_enc.fit(
	images_train_norm, 
	images_dev_norm, 
	images_test_norm, 
 	n_epochs=30, batch_size=100)

auto_enc.save_parameters(args.save_model_dir+'auto_enc')

if args.visualize:
	index = random.randint(0, images_test.shape[0])
	image = images_test[index]

	visualize_image(image, name="Original", resize=True)

	image_norm = std_normalize(np.array([image]))

	reconstructed = auto_enc.reconstruct(image_norm)

	reconstructed = (reconstructed[0]*image.std()) + image.mean()
	reconstructed.astype(np.uint8)
	visualize_image(reconstructed, name="Reconstruction", resize=True)








