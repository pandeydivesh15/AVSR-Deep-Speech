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

### Installing

* Firstly, install [Git Large File Storage(LFS) Support](https://git-lfs.github.com/).
* Open terminal and type following commands.
```bash
$ git clone https://github.com/pandeydivesh15/AVSR-Deep-Speech.git
$ cd AVSR-Deep-Speech
$ pip install -r requirements.txt 
```











