#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=0

########## Train Transformer VC model ##########

# Config 1:
# Speaker Embed = notatinput
# pitch = no
python main.py --config /mnt/data1/waris/repo/vc-spk-loss/conf/transformer_vc_ppg2mel.yaml \
               --name=transformer-vc \
               --seed=2 \
               --transformervc
#
# Status: Running
#