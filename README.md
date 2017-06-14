# Audio and Visual Speech Recognition(AVSR) using Deep Learning

_This is my [Google Summer of Code 2017](https://summerofcode.withgoogle.com/projects/#5019227963523072) Project with [the Distributed Little Red Hen Lab](http://www.redhenlab.org/)._

The aim of this project is to develop a working Speech to Text module for the Red Hen Lab’s current [Audio processing pipeline](https://github.com/RedHenLab/Audio). The initial goal is to extend current [Deep Speech model](https://github.com/mozilla/DeepSpeech)(audio only) to Red Hen lab's TV news videos datasets.

Now, it is common for news videos to incorporate both auditory and visual modalities. Developing a multi-modal Speech to Text model seems very tempting for these datasets. The next goal is to develop a multi-modal Speech to Text system (AVSR) by extracting visual modalities and concatenating them to the previous inputs.

This project is based on the approach discussed in paper [Deep Speech](https://arxiv.org/abs/1412.5567). This paper discusses speech recognition using audio modality only, hence this project can be seen as an extension to Deep Speech model.

## Getting Started

### Prerequisites

* [Git Large File Storage](https://git-lfs.github.com/)
* [TensorFlow 1.0 or above](https://www.tensorflow.org/install/)
* [SciPy](https://scipy.org/install.html)
* [PyXDG](https://pypi.python.org/pypi/pyxdg)
* [python_speech_features](https://pypi.python.org/pypi/python_speech_features)
* [python sox](https://pypi.python.org/pypi/sox)
* [pandas](https://pypi.python.org/pypi/pandas#downloads)
* [FFmpeg](https://www.ffmpeg.org/download.html)

### Installing

* Firstly, install [Git Large File Storage(LFS) Support](https://git-lfs.github.com/) and [FFmpeg](https://www.ffmpeg.org/download.html).
* Open terminal and type following commands.
```bash
$ git clone https://github.com/pandeydivesh15/AVSR-Deep-Speech.git
$ cd AVSR-Deep-Speech
$ pip install -r requirements.txt 
```

### Data-Preprocessing for Training

Please note that these data-preprocessing steps are only required if your training audio/video files are quite long (> 1 min). **If you have access to shorter wav files (length in secs) and their associated transcripts, you will not require any data-preprocessing** (you will need CSV files too, see [bin/import_ldc93s1.py](./bin/import_ldc93s1.py) for one example). In case you have longer audio/video files, it is suggested to use data-preprocessing.

These steps require videos/audios and their associated time-aligned transcripts. Time aligned time stamps for your audios/videos can be found using [Gentle](https://github.com/lowerquality/gentle/) or [Red Hen Lab's Audio Pipeline](https://github.com/RedHenLab/Audio/tree/master/Pipeline) or any other alignment application.

Store time-aligned timescripts as json files. The json file should be of the format: [Click here](https://gist.github.com/pandeydivesh15/2012ab10562cc85e796e1f57554aca33).

*Note*: By default, the project assumes that all .mp4(video) files are kept at [data/RHL_mp4](./data/RHL_mp4), json files at [data/RHL_json](./data/RHL_json) and all wav files at [data/RHL_wav](./data/RHL_wav). If you would like to change the defaults, change the associated variables at [bin/preprocess_data.py](./bin/preprocess_data.py).

#### Audio-only Speech Recognition

[bin/preprocess_data.py](./bin/preprocess_data.py) expects 5 positional arguments.

Argument			|	Less 
---					|	---
output_dir_train	|	Output dir for storing training files (with trailing slash)
output_dir_dev		|	Output dir for storing files for validation (with trailing slash)
output_dir_test		|	Output dir for storing test files (with a trailing slash)
train_split			|	A float value for deciding percentage of data split for training the model
dev_split			|	A float value for deciding percentage of validation data
test_split			|	A float value for deciding percentage of test data

Have a look at [./bin/preprocess_audio.sh](./bin/preprocess_audio.sh), for a sample usage. This script runs [bin/preprocess_data.py](./bin/preprocess_data.py) with default storage locations and default data split percentages. 

From the main project's directory, open terminal and type:

```bash
$ ./bin/preprocess_audio.sh
```

After this step, all prepared data files(train, dev, test) will be stored in data/clean_data folder.

### Training your model















