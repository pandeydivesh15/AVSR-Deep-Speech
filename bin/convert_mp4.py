import glob
import subprocess

audio_dir = "../data/RHL_wav/"
video_dir = "../data/RHL_mp4/"

# Get all file names
video_file_names = sorted(glob.glob(video_dir + "*.mp4"))
# Extract actual names of file, also remove any extensions
video_names = map(lambda x : x.split('/')[-1].split(".")[0], video_file_names)

# Command for converting video to audio
command = "ffmpeg -i " + video_dir + "{0}.mp4 -ab 96k -ar 44100 -vn " + audio_dir + "{0}.wav"

for name in video_names:
    subprocess.call(command.format(name), shell=True)