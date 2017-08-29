# Instructions for running project at CASE HPC

Please read the main project's README before proceeding here.

## Prerequisites

* First, copy project's code in your directory.
```bash
$ cp -r /home/dxp329/AVSR_Deep_Speech/AVSR-Deep-Speech ./AVSR-Deep-Speech
```

* If you like, you can also copy the singularity image from my directory.
```bash
$ cp /home/dxp329/rh_avsr_20170720.img ./
```

* Start singularity shell
```
$ module load singularity/2.3.1
$ singularity shell --nv rh_avsr_20170720.img
```

* Some of the python requirements are not present in the singularity image, so kindly install them manually.
```
$ cd AVSR-Deep-Speech
$ pip install --user -r requirements.txt
$ pip install --user scikit-video==0.1.2
```
**Note**: If there are more missing python dependencies later, please install them manually.

## Running the code

**Make sure you are in the project's dir for all steps.**

```$ cd AVSR-Deep-Speech``` 

Before continuing, copy some videos and corresponding json files to correct locations (required by the project). You can do so by running this script.
```bash
python bin/copy_videos_and_json.py path_to_mp4/ path_to_JSON/ n_videos_to_be_used
```

Some files are present in my home directory. Try running this command.
```bash
python bin/copy_videos_and_json.py /home/dxp329/videos/ /home/dxp329/json/ 1
```

### Audio-only Speech recognition.

#### Data Preprocessing

```bash
$ ./bin/preprocess_audio.sh
```

#### Training
```bash
$ ./bin/run_case_HPC.sh
```

#### Running exported model (trained model)
```bash
python ./bin/run_exported_model_audio.py -d path_to_exported_model/ -n model_name -af /path_to_wav_file/file.wav --use_spell_check 
```

One exported model is present at ```data/exports/export_/00000001/```. Use the model by:
```bash
python ./bin/run_exported_model_audio.py -d data/exports/export_/00000001/ -n export -af /path_to_wav_file/file.wav --use_spell_check
```

### Audio-Visual Speech Recognition

#### Data Preprocessing

1. First, create data for training RBMs and Autoencoder
```bash
$ ./bin/preprocessing_AE.sh
```

2. Train RBMs and autoencoder
```bash
$ ./bin/run_AE_training.sh
```

3. Prepare data for AVSR
```bash
$ ./bin/preprocess_AVSR.sh
```

#### Training

Before beginning training, run ```$ python bin/data_cleaner_CTC.py``` once. Read comments in bin/data_cleaner_CTC.py for more details.

```bash
$ ./bin/run_case_HPC_AVSR.sh
```

#### Running exported model (trained model)
```bash
python ./bin/run_exported_model_AVSR.py -d path_to_exported_model/ -n model_name -vf /path_to_video_file/file.mp4 
```

One exported model (not properly trained) is present at ```data/exports/export_AVSR/00000001/```. Use the model by:
```bash
python ./bin/run_exported_model_audio.py -d data/exports/export_AVSR/00000001/ -n export -af /path_to_video_file/file.mp4 --use_spell_check
```

### Important points 

* If you decide to retrain some model (with some new parameters/architecture), make sure that you remove all the previous checkpoints (made by previous model). DeepSpeech_RHL.py / DeepSpeech_RHL_AVSR.py tries to use the checkpoints if they present. Also, delete previous exports if there were any.
```
rm -r ~/.local/share/deepspeech/
rm -r data/export/
```
* If you decide to use data preprocessing multiple times, ensure that you remove any previous processed data.
```
rm -r data/clean_data/
rm -r data/clean_data_AVSR/
```
* While running exported models(both cases), try to give short length audio/video files for better/faster results.
* **Feel free to modify any bash scripts used above.**





