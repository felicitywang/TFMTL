#!/usr/bin/env bash
set -e

mkdir -p data/raw/SSTb
mkdir -p data/json/SSTb
mkdir -p data/raw/RTU
mkdir -p data/json/RTU

echo "Downloading the SSTb data..."
wget -nc https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

unzip trainDevTestTrees_PTB.zip
mv -f trainDevTestTrees_PTB.zip data/raw/SSTb/

echo "Converting the SSTb data to json..."
python3 ../../datasets/sentiment/SSTb/convert_SSTb_to_JSON.py ./
mv -f trees data/raw/SSTb/
mv -f data.json.gz data/json/SSTb/
mv -f index.json.gz data/json/SSTb/

echo "Downloading the RTU data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/rtu.tar.gz

echo "Untarring the RTU data..."
tar -zxvf rtu.tar.gz
mv -f rtu.tar.gz data/raw/RTU/
#rm -fr aclImdb_v1.tar.gz

echo "Converting the RTU data to json..."
python3 ../../datasets/sentiment/RTU/convert_RTU_to_JSON.py ./

mv -f rtu/ data/raw/RTU/
mv -f data.json.gz data/json/RTU/
mv -f index.json.gz data/json/RTU/

mkdir -p data/tf/merged
echo "Generating TFRecord files..."
python ../scripts/write_tfrecords_merged.py SSTb RTU
