# Audio and Visual Speech Recognition(AVSR) using Deep Learning

_This is my [Google Summer of Code 2017](https://summerofcode.withgoogle.com/projects/#5019227963523072) Project with [the Distributed Little Red Hen Lab](http://www.redhenlab.org/)._

The aim of this project is to develop a working Speech to Text module for the Red Hen Lab’s current [Audio processing pipeline](https://github.com/RedHenLab/Audio). The initial goal is to extend current [Deep Speech model](https://github.com/mozilla/DeepSpeech)(audio only) to Red Hen lab's TV news videos datasets.

Now, it is common for news videos to incorporate both auditory and visual modalities. Developing a multi-modal Speech to Text model seems very tempting for these datasets. The next goal is to develop a multi-modal Speech to Text system (AVSR) by extracting visual modalities and concatenating them to the previous inputs.

This project is based on the approach discussed in paper [Deep Speech](https://arxiv.org/abs/1412.5567). This paper discusses speech recognition using audio modality only, hence this project can be seen as an extension to Deep Speech model.

## Project Structure

1. **bin** directory: Contains all helper scripts. A fully developed project will be run through these scripts only.

2. **data**: All media files and CSV files reside here. Contains no code.

3. **util**: All helper Python classes and other scripts. These scripts contains:

	* Dataset handling Classes: The data stored in CSV files and media files handled here. The neural architecture interacts with this class(or classes) for data inflow.

	* Spell Check script: The spell checking system(KenLM) resides here.

	* Text/Audio/Video Handling scripts: Reading/analyzing text/audio/video, calculating MFCC frames, etc.

4. **AVSRDeepSpeech.py**: Main script of the project. Constructs our neural architecture, trains the model, and exports trained weights. This script is same as primary Deep Speech's script (very few changes).

## Getting Started

### Prerequisites

* [TensorFlow 1.0 or above](https://www.tensorflow.org/install/)
* [SciPy](https://scipy.org/install.html)








