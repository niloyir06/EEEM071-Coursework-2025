#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate pytorch

MODEL=resnet50 # mobilenet_v3_small resnet50 resnet50_fc512 vgg16 efficientnet_v2_s
OPTIMIZER=amsgrad
LR=0.0003
BATCH_SIZE=64
AUGMENTS=none
SAVE_DIR="$HOME/EEEM071/coursework/logs/${MODEL}_${OPTIMIZER}_${BATCH_SIZE}_${LR}_${AUGMENTS}"

STUDENT_ID=6902608 STUDENT_NAME="Niloy Irtisam" python main.py \
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
--train-batch-size $BATCH_SIZE \
--test-batch-size 100 \
--save-dir $SAVE_DIR \
#--random-erase \
#--color-jitter \
#--color-aug  \
