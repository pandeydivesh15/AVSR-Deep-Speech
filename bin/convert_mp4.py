import glob
import subprocess

def convert_mp4(video_dir, audio_dir):
	# Get all file names
	video_file_names = sorted(glob.glob(video_dir + "*.mp4"))
	# Extract actual names of file, also remove any extensions
	video_names = map(lambda x : x.split('/')[-1].split(".")[0], video_file_names)

	# Command for converting video to audio
	command = "ffmpeg -i " + video_dir + "{0}.mp4 -ab 96k -ar 44100 -vn " + audio_dir + "{0}.wav"

	for name in video_names:
	    subprocess.call(command.format(name), shell=True)

if __name__ == '__main__':
	# This part will only run from main project's directory.
	# If ran from any other dir, no error is shown.
	audio_dir = "data/RHL_wav/"
	video_dir = "data/RHL_mp4/"
	convert_mp4(video_dir, audio_dir)