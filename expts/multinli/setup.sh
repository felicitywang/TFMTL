#!/usr/bin/env bash

set -e

mkdir -p data/raw/MultiNLI
mkdir -p data/json/MultiNLI

echo "Downloading the MultiNLI data..."
# Download the Multi-NLI dataset
wget http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip
unzip multinli_0.9.zip -d multinli_0.9
rm multinli_0.9.zip
mv -f multinli_0.9 data/raw/MultiNLI

echo "Converting the MultiNLI data to json..."
# Downsample MultiNLI to 10K examples (according to Isabelle Augenstein, personal correspondence)
python3 ../../datasets/nli/MultiNLI/convert_MultiNLI_to_JSON.py ./data/raw/MultiNLI/ True 10000
mv -f data/raw/MultiNLI/data.json.gz data/json/MultiNLI/
mv -f data/raw/MultiNLI/index.json.gz data/json/MultiNLI/

cp -f ../../datasets/nli/MultiNLI/label.json ./data/json/MultiNLI/label.json

echo "Generating TFRecord files..."
python ../scripts/write_tfrecords_single.py MultiNLI
