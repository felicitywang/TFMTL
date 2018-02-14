#! /usr/bin/env bash

set -e

PATH1=data/tf/merged/min_50_max_-1/LMRD/
PATH2=data/tf/merged/min_50_max_-1/SSTb/
PATHS="$PATH1 $PATH2"
VOCAB=data/tf/merged/min_50_max_-1/vocab_size.txt

./gen_trainer.py --datasets IMDB SSTb \
                 --dataset_paths $PATHS \
                 --vocab_path $VOCAB \
                 "$@"
