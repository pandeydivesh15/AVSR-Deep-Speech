import os
import glob
import random 

import json
import tqdm
import cv2
import dlib
import tensorflow as tf
import numpy as np

from util.data_preprocessing import read_json_file, find_text_and_time_limits, split
from util.data_preprocessing_autoencoder import resize, get_mouth_coord
from util.video_stream import VideoStream 
from util.autoencoder import AutoEncoder

FACE_DETECTOR_MODEL = None
LANDMARKS_PREDICTOR = None

IMAGE_WIDTH = 500 # Every frame will be resized to this width before any processing

VIDEO_DIR = "./data/RHL_mp4/"
AUDIO_DIR = "./data/RHL_wav/"
JSON_DIR  = "./data/RHL_json/"
AE_MODEL_DIR = "./data/AE_and_RBM_model_saves/"
AE_LAYER_NAMES = [['gbrbm_1_w', 'gbrbm_1_h'],
				  ['bbrbm_1_w', 'bbrbm_1_h'],
				  ['bbrbm_2_w', 'bbrbm_2_h'], 
				  ['bgrbm_1_w', 'bgrbm_1_h']]

PROCESSED_VIDEO_DATA_DIR = "./data/auto_encoder_output/" 

AUTO_ENCODER = None

def load_trained_models():
	if not os.path.isfile("data/dlib_data/shape_predictor_68_face_landmarks.dat"):
		return
	global FACE_DETECTOR_MODEL, LANDMARKS_PREDICTOR

	FACE_DETECTOR_MODEL = dlib.get_frontal_face_detector()
	LANDMARKS_PREDICTOR = dlib.shape_predictor("data/dlib_data/shape_predictor_68_face_landmarks.dat")

def load_AE():
	global AUTO_ENCODER

	if AUTO_ENCODER is not None:
		return

	AUTO_ENCODER = AutoEncoder(
					32*32, 
					encoding_layer_sizes=[2000, 1000, 500, 50], 
					layer_names=AE_LAYER_NAMES)

	AUTO_ENCODER.load_parameters(AE_MODEL_DIR+'auto_enc')

def crop_suitable_face(rects, frame):
	if not rects:
		# If NO face is present in the frame, return `None`
		return None
	rect = rects[0]
	if len(rects) > 1:
		# TODO
		# If there are mutliple faces detected, select suitable frame.
		# For now, we choose the first face in the `rects`. 
		# Implementation left. 
		rect = rects[0]

	# Get landmarks
	landmarks = LANDMARKS_PREDICTOR(frame, rect)
	mouth_coordinates = get_mouth_coord(landmarks)

	# Crop suitable portion
	x, y, w, h = cv2.boundingRect(mouth_coordinates)
	mouth_roi = frame[y:y + h, x:x + w]

	h, w, channels = mouth_roi.shape
	# If the cropped region is very small, ignore this case.
	if h < 10 or w < 10:
		return None
	
	resized = resize(mouth_roi, 32, 32)
	resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	return resized

def validate_frames(all_frames):	
	mouth_regions = []
	for frame in all_frames:
		rects = FACE_DETECTOR_MODEL(frame, 0)
		region = crop_suitable_face(rects, frame)
		if region is None:
			return None
		region = region.reshape(32*32)
		mouth_regions.append(region)

	return np.array(mouth_regions)

def run_video_and_refine(video_file_path, split_info):
	video_name = video_file_path.split('/')[-1].split(".")[0]

	stream = VideoStream(video_file_path)
	stream.start()

	FPS = stream.stream.get(cv2.CAP_PROP_FPS)

	time_elapsed = 0.00
	time_end = split_info[-1][1][1] 

	# `split_info` is a list of tuples of the form (x, (y, z))
	frame_count = 0

	data = []
	
	for info, i in zip(split_info, range(len(split_info))):
		# `info` is tuple of the form (x, (y, z)), y = split_time_start, z = split_time_end
		# Please refer to util.data_preprocessing.find_text_and_time_limits() for more details.
		while time_elapsed < info[1][0]:
			frame = stream.read()
			frame_count += 1

			time_elapsed = frame_count*(1.00/FPS)

		# This section of code does actual preprocessing
		all_frames = []

		while time_elapsed <= info[1][1]:
			frame = stream.read()
			frame = resize(frame, IMAGE_WIDTH)
			frame_count += 1
			all_frames.append(frame)

			time_elapsed = frame_count*(1.0000/FPS)

		mouth_regions = validate_frames(all_frames)

		if mouth_regions is not None:
			split_file_name = video_name + str(i).zfill(5)
			data.append((split_file_name, video_name, mouth_regions, info))	
			
	stream.stop()
	return data

def encode_and_store(batch_x, output_dir, file_name):
	global AUTO_ENCODER
	if AUTO_ENCODER is None:
		load_AE()

	norm_batch = np.zeros(batch_x.shape)
	for i in range(len(batch_x)):
		norm_batch[i] = (batch_x[i] - np.mean(batch_x[i])) / np.std(batch_x[i])

	output_dict = {
		'name' : file_name,
		'encoded': AUTO_ENCODER.transform(norm_batch).tolist()}

	with open(output_dir+file_name+'.json', 'w') as f:
		json.dump(output_dict, f)

def preprocess_videos(output_dir_train, output_dir_dev, output_dir_test, 
					  train_split, dev_split, test_split):
	json_file_paths = sorted(glob.glob(JSON_DIR + "*.json"))

	load_trained_models()

	split_info = []
	for file_path in json_file_paths:
		data = read_json_file(file_path)
		split_info.append(find_text_and_time_limits(data))

	data = []
	for path, info in zip(json_file_paths, split_info):
		file_name = path.split('/')[-1].split('.')[0]
		data.extend(run_video_and_refine(VIDEO_DIR+file_name+'.mp4', info))

	random.shuffle(data)

	total_split_files = len(data)
	dev_limit_start = int(train_split * total_split_files)
	dev_limit_end = dev_limit_start + int(dev_split * total_split_files)

	split_file_count = 0 # Counts number of split files.
	# Hence helps in data split between train/dev/test dir.

	for i in tqdm.tqdm(data):
		# Set output directory either equal to train or test or dev
		# Decided by args `train_split`, `dev_split`, `test_split`
		if split_file_count < dev_limit_start:
			output_dir = output_dir_train
		elif split_file_count < dev_limit_end:
			output_dir = output_dir_dev
		else:
			output_dir = output_dir_test


		# Encode all the mouth region images using AutoEncoder and store encodings
		# For visual speech recognition
		encode_and_store(i[2], output_dir, i[0])

		# Now, split and store audio according to time info
		# For audio speech recognition
		split(
			split_file_path=output_dir+i[0]+'.wav',
			main_file_path=AUDIO_DIR+i[1]+'.wav',
			transcript_path=output_dir+i[0]+'.txt',
			split_info=i[3])

		split_file_count += 1

def extract_and_store_visual_features(video_file_path, json_dir, json_name):
	load_trained_models()
	load_AE()

	stream = VideoStream(video_file_path)
	stream.start()

	mouth_regions = []
	cnt = 0
	while not stream.is_empty():
		frame = stream.read()
		frame = resize(frame, IMAGE_WIDTH)
		rects = FACE_DETECTOR_MODEL(frame, 0)
		region = crop_suitable_face(rects, frame)
		if region is None:
			# If no proper face region could be detected, we fill normally distributed random values
			mouth_regions.append(np.random.normal(size=32*32))
		else:
			mouth_regions.append(region.reshape(32*32))

	mouth_regions = np.array(mouth_regions)
	
	encode_and_store(mouth_regions, json_dir, json_name.split('.')[0])
	AUTO_ENCODER.close()









