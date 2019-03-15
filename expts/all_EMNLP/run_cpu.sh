#! /usr/bin/env bash

cd /export/a08/fwang/tfmtl/expts/all_EMNLP
source /home/fwang/cpu/bin/activate

programname=$0

function usage {
    echo "usage: $programname HPARAMS"
    exit 1
}

# Config
SAVE_SECS=120

# Paths
EXPT=expts/Stance
VOCAB=all_0
CORPUS=data/tf/Stance-mt/$VOCAB

STANCE=$CORPUS/Stance
FNC=$CORPUS/FNC-1
MULTINLI=$CORPUS/MultiNLI
TARGET=$CORPUS/Target

echo "Hyper-parameters: $1"
TIME=`date +%Y-%m-%d_%H-%M-%S`

if [ ! -d "expts" ]; then
    mkdir expts
fi

if [ ! -d "$EXPT" ]; then
    mkdir $EXPT
fi


cd /export/a08/fwang/tfmtl/expts/all_EMNLP
mkdir $EXPT/$TIME
python /export/a08/fwang/tfmtl/expts/all_EMNLP/task.py \
       --job-dir $EXPT/$TIME \
       --train-corpora $STANCE $FNC $MULTINLI $TARGET \
       --eval-corpus $STANCE \
       --save-secs $SAVE_SECS \
       --hparams "$1"  # &> $EXPT/$TIME/log.txt


# eos
