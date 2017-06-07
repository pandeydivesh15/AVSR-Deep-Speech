import glob
import wave
import json
import os
import pandas

WAV_FILE_MIN_LEN = 2.00 # in seconds

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

def split(split_info_tuple, file_name, file_path, output_dir):
	audio_file = wave.open(file_path, 'rb')

	for data, i  in zip(split_info_tuple, range(len(split_info_tuple))):
		split_file_name = output_dir + file_name + str(i).zfill(5) + ".wav"
		split_file = wave.open(split_file_name, 'wb')

		t0, t1 = data[1] # cut audio between t0, t1 seconds
		s0, s1 = int(t0*audio_file.getframerate()), int(t1*audio_file.getframerate())
		
		# The next step forces me to open and close the main file multiple times, in order to prepare correct data. 
		audio_file.readframes(s0) # discard frames upto s0
		frames = audio_file.readframes(s1-s0)

		split_file.setparams(audio_file.getparams())
		split_file.writeframes(frames)
		split_file.close()
		

		# Store transcript
		with open(output_dir + file_name + str(i).zfill(5) + ".txt", 'wb') as f:
			f.write(data[0])

		# TODO: Get rid of multiple opening and closing of the same main audio file.
		audio_file.close()
		audio_file = wave.open(file_path, 'rb')

def split_aligned_audio(audio_dir, json_dir	, output_dir):
	json_file_names = sorted(glob.glob(json_dir + "*.json?"))

	for file_path in json_file_names:
		data = read_json_file(file_path)
		split_info = find_text_and_time_limits(data)

		audio_file_name = file_path.split('/')[-1].split('.')[0]
		audio_file_path = audio_dir + audio_file_name + '.wav'

		split(split_info, audio_file_name, audio_file_path, output_dir)


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
