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
from util.data_preprocessing_autoencoder import resize, get_mouth_coord, visualize
from util.video_stream import VideoStream 
from util.autoencoder import AutoEncoder

FACE_DETECTOR_MODEL = None
LANDMARKS_PREDICTOR = None

IMAGE_WIDTH = 500 # Every frame will be resized to this width before any processing

VIDEO_DIR = "./data/RHL_mp4/"
AUDIO_DIR = "./data/RHL_wav/"
JSON_DIR  = "./data/RHL_json/"
AE_MODEL_DIR = "./data/AE_and_RBM_model_saves/"
AE_LAYER_NAMES = [['gbrbm_1_w', 'gbrbm_1_h'], # Layer names will help us in restoring trained Autoencoder
				  ['bbrbm_1_w', 'bbrbm_1_h'],
				  ['bbrbm_2_w', 'bbrbm_2_h'], 
				  ['bgrbm_1_w', 'bgrbm_1_h']] 

AUTO_ENCODER = None

def load_trained_models():
	if not os.path.isfile("data/dlib_data/shape_predictor_68_face_landmarks.dat"):
		return
	global FACE_DETECTOR_MODEL, LANDMARKS_PREDICTOR

	FACE_DETECTOR_MODEL = dlib.get_frontal_face_detector()
	LANDMARKS_PREDICTOR = dlib.shape_predictor("data/dlib_data/shape_predictor_68_face_landmarks.dat")

def load_AE():
	"""
		Loads trained autoencoder (for finding encoded features for each mouth region).
	"""
	global AUTO_ENCODER

	if AUTO_ENCODER is not None:
		return

	AUTO_ENCODER = AutoEncoder(
					32*32, 
					encoding_layer_sizes=[2000, 1000, 500, 50], 
					layer_names=AE_LAYER_NAMES)

	AUTO_ENCODER.load_parameters(AE_MODEL_DIR+'auto_enc')

def crop_suitable_face(rects, frame, prev_frame_faces=None):
	"""
	Args:
		1. rects:				all faces detected in a given frame by `FACE_DETECTOR_MODEL`
		2. frame:				current frame of the video.
		3. prev_frame_faces:	If given, speaker detection (among multiple faces) will be used.
								Every element of `prev_frame_faces` must be a tuple (x, y); 
								x = rects found in previous frame.
								y = previous frame
								Length of `prev_frame_faces` will determine how many previous frames to be used.
	"""

	if not rects:
		# If NO face is present in the frame, return some suitable face from `prev_frame_faces`.
		# IF no suitable face found, return `None`
		if not prev_frame_faces:
			# If no previous frame
			return None
		# Recursive definition	
		return crop_suitable_face(prev_frame_faces[-1][0], prev_frame_faces[-1][1], prev_frame_faces[:-1])

	mouth_coordinates = None 
	# We need to find correct mouth region in the frame: Speaker's mouth region

	if len(rects) > 1:
		# If there are multiple faces detected, select suitable frame.
		if (not prev_frame_faces) or (len(prev_frame_faces[-1][0]) != len(rects)):
			# If no previous frames info available OR 
			# If faces detected in current frame were not present in the previous frame,
			# 	(currently only checking if number of faces in current and previous frame are not equal)
			# Then, we select most probable face in the current frame, 
			# We choose the face whose mouth is wide open (largest area in the frame).

			max_mouth_area = -1.0
	
			for r in rects:
				landmarks = LANDMARKS_PREDICTOR(frame, r)
				coordinates = get_mouth_coord(landmarks)
				area = cv2.contourArea(coordinates)
				if max_mouth_area < area:
					max_mouth_area = area
					mouth_coordinates = coordinates
		else:
			# Select best speaker face by calculating mean squared difference
			# of the pixel values (mouth region) between the current and the previous frame.

			# It is expected that both current frame's `rects` and previous frame's `rects`
			# contains same faces and in same order.

			max_square_diff = -1.0

			for curr_r, prev_r in zip(rects, prev_frame_faces[-1][0]):
				landmarks = LANDMARKS_PREDICTOR(frame, curr_r)
				coordinates = get_mouth_coord(landmarks)				
				# Find bounding rectangle for mouth coordinates
				x, y, w, h = cv2.boundingRect(coordinates)				
				
				mouth_roi_1 = frame[y:y + h, x:x + w]
				mouth_roi_2 = prev_frame_faces[-1][1][y:y + h, x:x + w]
				
				MSD = ((mouth_roi_1 - mouth_roi_2)**2).mean()
				# MSD = Mean Square Difference
				if MSD > max_square_diff:
					max_square_diff = MSD
					mouth_coordinates = coordinates
					
	else:
		# If there is only one face.
		# Get landmarks
		landmarks = LANDMARKS_PREDICTOR(frame, rects[0])
		mouth_coordinates = get_mouth_coord(landmarks)

	# Crop suitable portion
	x, y, w, h = cv2.boundingRect(mouth_coordinates)
	mouth_roi = frame[y:y + h, x:x + w]

	h, w, channels = mouth_roi.shape
	# If the cropped region is very small, ignore this case.
	if h < 4 or w < 4:
		return None
	
	resized = resize(mouth_roi, 32, 32)
	resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	return resized

def validate_frames(all_frames):
	"""
	Args:
		1. all_frames:		All frames to be studied for training our AVSR model.

	If this function returns `None`, then `all_frames` are not suitable for training our AVSR.
	In other cases, the function returns numpy array of mouth regions.
	"""	
	prev_frame_faces = []
	# Keeps record of previous frames and faces found in previous frames. Initially it is empty.
	# This will help in speaker identification in case of multiple faces detected.

	mouth_regions = []
	for frame in all_frames:
		rects = FACE_DETECTOR_MODEL(frame, 0)
		region = crop_suitable_face(rects, frame, prev_frame_faces)
		if region is None:
			return None
		
		# Update previous frames. Max size of `prev_frame_faces` = 5
		prev_frame_faces.append((rects, frame))
		if len(prev_frame_faces) > 5 : prev_frame_faces.remove(prev_frame_faces[0])

		region = region.reshape(32*32)
		mouth_regions.append(region)

	return np.array(mouth_regions)

def run_video_and_refine(video_file_path, split_info):
	"""
	Args:
		1. video_file_path:	File path to be used for training AVSR.
		2. split_info:		This variable is similar to the one used in the case of audio-only model training.
							For more details, please see util.data_preprocessing.find_text_and_time_limits() 
							and util.data_preprocessing..split() functions.

	"""
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
	"""
	Args:
		1. batch_x:			Batch of 32*32 images which will go inside our autoencoder.
		2. output_dir:		Dir path for storing all encoded features for given `batch_x`.
							Features will be stored in the form of JSON file.
		3. file_name:		File name of JSON file.
	"""
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
	"""
	Args:
		1. output_dir_train:	Dir which will store all training files.
		2. output_dir_dev:		Dir which will store all validation files.
		3. output_dir_test:		Dir which will store all test files.
		4. train_split, dev_split, test_split:
								Ratio which decides number of training/validation/test files.
	"""
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
	"""
		This function extracts visual features from the given video and stores them in a JSON file.
		This function will NOT be used while preparing training/test/dev data for our model.
		This function only runs through `bin/run_exported_model_AVSR.py` script.
	Args:
		1. video_file_path:		File path for the video to be processed.
		2. json_dir:			Dir where JSON file will exist.
		3. json_name:			Name to be given to JSON file.
	"""
	# first load all trained models: DLIB's models and Autoencoder.
	load_trained_models()
	load_AE()

	# Start reading video.
	stream = VideoStream(video_file_path)
	stream.start()

	mouth_regions = []
	prev_frame_faces = []
	# Keeps record of previous frames and faces found in previous frames. Initially it is empty.

	while not stream.is_empty():
		frame = stream.read()
		frame = resize(frame, IMAGE_WIDTH)
		rects = FACE_DETECTOR_MODEL(frame, 0)

		# rects = FACE_DETECTOR_MODEL(frame, 1) 
		# This can increase efficiency in detecting mutiple faces; this can help in correct speaker detection.
		# However, computing time will also be increased

		region = crop_suitable_face(rects, frame, prev_frame_faces)
		if region is None:
			# If no proper face region could be detected, we fill normally distributed random values
			mouth_regions.append(np.random.normal(size=32*32))
		else:
			mouth_regions.append(region.reshape(32*32))

		# Update previous frames. Max size of `prev_frame_faces` = 5
		prev_frame_faces.append((rects, frame))
		if len(prev_frame_faces) > 5 : prev_frame_faces.remove(prev_frame_faces[0])

	mouth_regions = np.array(mouth_regions)
	
	# Find and store visual features.
	encode_and_store(mouth_regions, json_dir, json_name.split('.')[0])
	AUTO_ENCODER.close()









