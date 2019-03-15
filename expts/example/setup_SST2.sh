#!/usr/bin/env bash
set -e

mkdir -p data/raw/SST2
mkdir -p data/json/SST2


echo "Downloading the SSTb data..."
wget -nc https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

unzip trainDevTestTrees_PTB.zip
mv -f trainDevTestTrees_PTB.zip data/raw/SST2/

echo "Converting the SST2 data to json..."
python3 ../../datasets/sentiment/SST2/convert_SST2_to_JSON.py ./
mv -f trees data/raw/SST2/
mv -f data.json.gz data/json/SST2/
mv -f index.json.gz data/json/SST2/

cp ../../datasets/sentiment/SST2/label.json ./label_SST2.json -f

