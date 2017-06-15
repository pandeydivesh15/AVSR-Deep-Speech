#!/bin/sh
set -xe
srun -p gpuk40 --gres=gpu:1 --pty bash
module load singularity
module load tensorflow
module load ffmpeg

singularity exec $TENSORFLOW $1