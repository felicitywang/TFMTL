#! /usr/bin/env bash

programname=$0

function usage {
    echo "usage: $programname HPARAMS"
    exit 1
}

# Config
SAVE_SECS=60

# Paths
EXPT=expts
DATA=data/tf/single/SSTb/min_1_max_-1

echo "Hyper-parameters: $HPS"
TIME=`date +%Y-%m-%d_%H-%M-%S`

if [ ! -d "$EXPT" ]; then
    mkdir $EXPT
fi

mkdir $EXPT/$TIME
python task.py \
       --job-dir $EXPT/$TIME \
       --train-file $DATA/train.tf \
       --eval-file $DATA/valid.tf \
       --save-secs $SAVE_SECS \
       --hparams "$HPS"  # &> $EXPT/$TIME/log.txt


# eos
