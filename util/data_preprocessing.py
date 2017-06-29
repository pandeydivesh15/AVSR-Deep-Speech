import glob
import wave
import json
import os
import pandas
import subprocess
import random

# Minimum length of split .wav file
WAV_FILE_MIN_LEN = 2.00 # in seconds
WAV_FILE_MAX_LEN = 10.00 # in seconds

def convert_mp4(video_dir, audio_dir):
	'''
	Args: 
		1. video_dir:	Directory for all video files
		2. audio_dir:	Directory where all converted files will be stored.
	'''

	# Get all file names
	video_file_names = sorted(glob.glob(video_dir + "*.mp4"))
	# Extract actual names of file, also remove any extensions
	video_names = map(lambda x : x.split('/')[-1].split(".")[0], video_file_names)

	# Command for converting video to audio
	command = "ffmpeg -i " + video_dir + "{0}.mp4 -ab 96k -ar 44100 -vn " + audio_dir + "{0}.wav"

	for name in video_names:
		subprocess.call(command.format(name), shell=True)

def read_json_file(file_path):
	'''
	Args:
		1. file_path:	File path for a json file. 
						File should be similar to the format -
						https://gist.github.com/pandeydivesh15/2012ab10562cc85e796e1f57554aca33
	Returns:
		data:	A list of dicts. Each dict contains timing info for a spoken word(or punctuation).
	'''

	with open(file_path, 'r') as f:
		data = json.loads(f.read())['words']

		# for line in f:
		# 	temp = json.loads(line)
		# 	temp['start'] = None if temp['start'] == 'NA' else float(temp['start'])
		# 	temp['end'] = None if temp['end'] == 'NA' else float(temp['end'])
		# 	try:
		# 		temp['word'] = temp['word'].encode('ascii')
		# 	except KeyError:
		# 		temp['punctuation'] = temp['punctuation'].encode('ascii')				
		# 	data.append(temp)

	return data

def find_text_and_time_limits(alignment_data):
	'''
	Args: 
		alignment_data:	The list of dicts returned by `read_json_file()` method.
						Contains time-alignment data for a specific .wav file.
	Returns:
		data:	A list with each element of the form (x, (y, z)). 
				Each element gives info for a new split .wav file.
				`x` represents transcript for a split wav file.
				`y` represents starting time (in secs) in actual .wav file.
				`z` represents ending time in actual .wav file.
	Note: Using `y` and `z` for each element in `data`, main .wav will be split into many parts.
	'''

	data = []
	# 'data' will hold tuples of the form (x, (y, z))

	# Determine starting time position for splitting main file
	count = 0 # Counts no. of initial values to skip in `alignment_data`
	for word_dict in alignment_data:
		if word_dict.has_key('start'):
			split_time_start = word_dict['start']
			break
		count += 1

	transcript  = ""

	# TODO: Find a way to get rid of three try and except statements.

	for word_dict in alignment_data[count:]:
		if word_dict['case'] == "not-found-in-audio":
			try:
				transcript += " " + word_dict['word'].encode('ascii')			
			except Exception:
				pass
			continue
		if word_dict['end'] - split_time_start >= WAV_FILE_MIN_LEN:
			try:
				transcript += " " + word_dict['word'].encode('ascii')			
			except Exception:
				pass
			data.append(
				(transcript.lower().strip(),
				(split_time_start, word_dict['end'])))

			transcript = ""
			split_time_start = word_dict['end']				
		else:
			try:
				transcript += " " + word_dict['word'].encode('ascii')			
			except Exception:
				pass
	return data

def split(split_file_path, main_file_path, transcript_path, split_info):
	'''
	Here, splitting takes place.

	Args:
		split_file_path:	File path for new split file.
		main_file_path:		File path for original .wav file.
		transcript_path:	File path where transcript will be written.
		split_info:			A tuple of the form (x, (y, z))
	'''
	audio_file = wave.open(main_file_path, 'rb')
	split_file = wave.open(split_file_path, 'wb')

	t0, t1 = split_info[1] # cut audio between t0, t1 seconds
	s0, s1 = int(t0*audio_file.getframerate()), int(t1*audio_file.getframerate())

	audio_file.readframes(s0) # discard frames up to s0
	frames = audio_file.readframes(s1-s0)

	split_file.setparams(audio_file.getparams())
	split_file.writeframes(frames)
	split_file.close()
	
	# Store transcript
	with open(transcript_path, 'wb') as f:
		f.write(split_info[0])

	# TODO: Get rid of multiple opening and closing of the same main audio file.
	audio_file.close()
	

def split_aligned_audio(audio_dir, json_dir	, output_dir_train, output_dir_dev,
						output_dir_test, train_split, dev_split, test_split):
	'''
	Args:
		audio_dir:			Dir where all main .wav file are stored.
		json_dir:			Dir where all related .json file are stored. 
							Json should exist for each video.
		output_dir_train:	Output train data dir.
		output_dir_dev:		Output validation data dir.
		output_dir_test:	Output test data dir.
		train_split:		Train data ratio / %
		dev_split:			Validation data ratio / %
		test_split:			Test data ratio / %	
	'''

	split_info = []
	total_audio_files = 0 # denotes total wav files we will get after splitting

	# Get all json files' paths, and calculate splitting info from each json file.
	json_file_names = sorted(glob.glob(json_dir + "*.json"))
	
	for file_path in json_file_names:
		data = read_json_file(file_path)
		split_info.append(find_text_and_time_limits(data))
		total_audio_files += len(split_info[-1])

	dev_limit_start = int(train_split * total_audio_files)
	dev_limit_end = dev_limit_start + int(dev_split * total_audio_files)

	split_file_count = 0 # Counts number of split files.
	# Hence helps in data split between train/dev/test dir.

	# Brings randomness in data --> shuffles all original audio filenamess and associated info  
	shuffled_data = zip(json_file_names, split_info)
	random.shuffle(shuffled_data)

	for file_path, info in shuffled_data:
		audio_file_name = file_path.split('/')[-1].split('.')[0]
		audio_file_path = audio_dir + audio_file_name + '.wav'

		# Brings randomness in data --> shuffles split info data for each original audio
		random.shuffle(info)

		for data, i  in zip(info, range(len(info))):
			# Set output directory either equal to train or test or dev
			# Decided by args `train_split`, `dev_split`, `test_split`
			if split_file_count < dev_limit_start:
				output_dir = output_dir_train
			elif split_file_count < dev_limit_end:
				output_dir = output_dir_dev
			else:
				output_dir = output_dir_test

			# Generate file paths for transcript and split .wav file.
			split_file_path = output_dir + audio_file_name + str(i).zfill(5) + ".wav"
			transcript_file_path = output_dir + audio_file_name + str(i).zfill(5) + ".txt"

			# Split the main file
			split(split_file_path, audio_file_path, transcript_file_path, data)
			split_file_count += 1


def create_csv(data_dir):
	'''
	Generates CSV file (as required by DeepSpeech_RHL.py) in the given dir.

	Args:
		data_dir:	Directory where all .wav files and 
					their associated timescripts are stored.

	'''

	# Get all audio and transcript file paths.
	audio_file_paths = sorted(glob.glob(data_dir + "*.wav"))
	transcript_file_paths = sorted(glob.glob(data_dir + "*.txt"))
	
	audio_file_sizes = []
	transcripts = []

	for x, y in zip(audio_file_paths, transcript_file_paths):
	    with open(y, "rb") as f:
	        transcripts.append(f.read())
	    
	    # Get file size.
	    metadata = os.stat(x)
	    audio_file_sizes.append(metadata.st_size)

	# Create pandas dataframe
	df = pandas.DataFrame(columns=["wav_filename", "wav_filesize", "transcript"])
	df["wav_filename"] = audio_file_paths
	df["wav_filesize"] = audio_file_sizes
	df["transcript"] = transcripts
	
	df.to_csv(data_dir + "data.csv", sep=",", index=None) # Save CSV
