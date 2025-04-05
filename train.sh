#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate pytorch

MODEL=mobilenet_v3_small
OPTIMIZER=amsgrad
LR=0.0003
SAVE_DIR="$HOME/EEEM071/coursework/logs/${MODEL}_${OPTIMIZER}_${LR}"

STUDENT_ID=kn00794 STUDENT_NAME="Jane Doe" python main.py \
-s veri \
-t veri \
-a $MODEL \
--root ../ \
--height 224 \
--width 224 \
--optim $OPTIMIZER \
--lr $LR \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir $SAVE_DIR