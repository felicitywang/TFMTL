#! /usr/bin/env bash

source /home/fwang/.bashrc
source /home/fwang/gpu/bin/activate
cd /export/a08/fwang/tfmtl/expts/all_EMNLP


programname=$0

function usage {
    echo "usage: $programname HPARAMS"
    exit 1
}

# Config
SAVE_SECS=30

# Paths
VOCAB=all_0

EXPT=/export/a08/fwang/tfmtl_data/GMTL-NAACL2019/results/Stance-st/all_0/Stance/GSTL_t
Stance=/export/a08/fwang/expts/tfmtl/all_EMNLP/data/tf/Stance-st/$VOCAB/Stance

echo "Hyper-parameters: $1"
TIME=`date +%Y-%m-%d_%H-%M-%S`

#if [ ! -d "expts" ]; then
#    mkdir expts
#fi

if [ ! -d "$EXPT" ]; then
    mkdir $EXPT
fi

mkdir $EXPT/$TIME
CUDA_VISIBLE_DEVICES=`free-gpu` python task_t.py \
       --job-dir $EXPT/$TIME \
       --train-corpora $Stance \
       --eval-corpus $Stance \
       --save-secs $SAVE_SECS \
       --device-type GPU \
       --hparams "$1"  # &> $EXPT/$TIME/log.txt


# eos
