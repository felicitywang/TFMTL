#!/usr/bin/env sh

# data to predict
mkdir -p data/json/predict/

# document level
cp /export/a05/mahsay/domain/data/json/doc -fr data/json/predict

# sentence level
cp /export/a05/mahsay/domain/data/json/sent -fr data/json/predict


cp /export/a05/mahsay/domain/data/json/doc/1B/EVAL3/tt20.bop/concat/data.json.gz data/json/predict/doc/1B/EVAL3/tt20.bop/concat 

