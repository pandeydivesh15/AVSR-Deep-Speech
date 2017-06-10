import glob
import wave
import json
import os
import pandas
import subprocess

WAV_FILE_MIN_LEN = 2.00 # in seconds

def convert_mp4(video_dir, audio_dir):
	# Get all file names
	video_file_names = sorted(glob.glob(video_dir + "*.mp4"))
	# Extract actual names of file, also remove any extensions
	video_names = map(lambda x : x.split('/')[-1].split(".")[0], video_file_names)

	# Command for converting video to audio
	command = "ffmpeg -i " + video_dir + "{0}.mp4 -ab 96k -ar 44100 -vn " + audio_dir + "{0}.wav"

	for name in video_names:
		subprocess.call(command.format(name), shell=True)

def read_json_file(file_path):
	data = []
	with open(file_path, 'r') as f:
		for line in f:
			temp = json.loads(line)
			temp['start'] = None if temp['start'] == 'NA' else float(temp['start'])
			temp['end'] = None if temp['end'] == 'NA' else float(temp['end'])
			try:
				temp['word'] = temp['word'].encode('ascii')
			except KeyError:
				temp['punctuation'] = temp['punctuation'].encode('ascii')				
			data.append(temp)

	return data

def find_text_and_time_limits(alignment_data):
	data = []
	# 'data' will hold items of the form (x, (y, z))
	split_time_start = 0.00
	transcript  = ""

	for word_dict in alignment_data:
		if word_dict['end'] is None:
			if word_dict.has_key('word'):
				transcript += " " + word_dict['word']
			continue
		if word_dict['end'] - split_time_start >= WAV_FILE_MIN_LEN:
			if word_dict.has_key("punctuation"):
				data.append((transcript.lower().strip(), (split_time_start, word_dict['end'])))
				transcript = ""
				split_time_start = word_dict['end']
			else:
				transcript += " " + word_dict['word']
		else:
			if word_dict.has_key('word'):
				transcript += " " + word_dict['word']
	return data

def split(split_file_path, main_file_path, transcript_path, split_info):
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
	split_info = []
	total_audio_files = 0 # denotes total wav files after splitting

	json_file_names = sorted(glob.glob(json_dir + "*.json?"))
	for file_path in json_file_names:
		data = read_json_file(file_path)
		split_info.append(find_text_and_time_limits(data))
		total_audio_files += len(split_info[-1])

	# TODO: Use a better way to split audios between train, dev, and test directories. Bring randomness
	dev_limit_start = int(train_split * total_audio_files)
	dev_limit_end = dev_limit_start + int(dev_split * total_audio_files)

	split_file_count = 0

	for file_path, info in zip(json_file_names, split_info):
		audio_file_name = file_path.split('/')[-1].split('.')[0]
		audio_file_path = audio_dir + audio_file_name + '.wav'

		for data, i  in zip(info, range(len(info))):
			# Set output directory either equal to train or test or dev
			if split_file_count < dev_limit_start:
				output_dir = output_dir_train
			elif split_file_count < dev_limit_end:
				output_dir = output_dir_dev
			else:
				output_dir = output_dir_test

			split_file_path = output_dir + audio_file_name + str(i).zfill(5) + ".wav"
			transcript_file_path = output_dir + audio_file_name + str(i).zfill(5) + ".txt"

			split(split_file_path, audio_file_path, transcript_file_path, data)
			split_file_count += 1


def create_csv(data_dir):
	audio_file_paths = sorted(glob.glob(data_dir + "*.wav"))
	transcript_file_paths = sorted(glob.glob(data_dir + "*.txt"))
	
	audio_file_sizes = []
	transcripts = []

	for x, y in zip(audio_file_paths, transcript_file_paths):
	    with open(y, "rb") as f:
	        transcripts.append(f.read())
	    
	    metadata = os.stat(x)
	    audio_file_sizes.append(metadata.st_size)

	df = pandas.DataFrame(columns=["wav_filename", "wav_filesize", "transcript"])
	df["wav_filename"] = audio_file_paths
	df["wav_filesize"] = audio_file_sizes
	df["transcript"] = transcripts
	
	df.to_csv(data_dir + "data.csv", sep=",", index=None)
