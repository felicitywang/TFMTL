#!/usr/bin/env bash
set -e

mkdir -p data/raw/SSTb
mkdir -p data/json/SSTb
mkdir -p data/raw/RTC
mkdir -p data/json/RTC

echo "Downloading the SSTb data..."
wget -nc https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

unzip trainDevTestTrees_PTB.zip
mv -f trainDevTestTrees_PTB.zip data/raw/SSTb/

echo "Converting the SSTb data to json..."
python3 ../../datasets/sentiment/SSTb/convert_SSTb_to_JSON.py ./
mv -f trees data/raw/SSTb/
mv -f data.json.gz data/json/SSTb/
mv -f index.json.gz data/json/SSTb/

cp ../../datasets/sentiment/SSTb/label.json ./label_SSTb.json -f

echo "Downloading the RTC data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/rtc.tar.gz

echo "Untarring the RTC data..."
tar -zxvf rtc.tar.gz
mv -f rtc.tar.gz data/raw/RTC/
#rm -fr aclImdb_v1.tar.gz

echo "Converting the RTC data to json..."
python3 ../../datasets/sentiment/RTC/convert_RTC_to_JSON.py ./

cp ../../datasets/sentiment/RTC/label.json ./label_RTC.json -f

mv -f rtc/ data/raw/RTC/
mv -f data.json.gz data/json/RTC/
mv -f index.json.gz data/json/RTC/

mkdir -p data/tf/merged
echo "Generating TFRecord files..."
python ../scripts/write_tfrecords_merged.py SSTb RTC
