#!/usr/bin/env bash
set -e

mkdir -p data/raw/IMDb
mkdir -p data/json/IMDb
mkdir -p data/raw/RTU
mkdir -p data/json/RTU

echo "Downloading the IMDb data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/imdb.tar.gz

tar zxvf imdb.tar.gz
mv -f imdb.tar.gz data/raw/IMDb/

echo "Converting the IMDb data to json..."
python3 ../../datasets/sentiment/IMDb/convert_IMDb_to_JSON.py ./
mv -f imdb data/raw/IMDb/
mv -f data.json.gz data/json/IMDb/
mv -f index.json.gz data/json/IMDb/

cp ../../datasets/sentiment/IMDb/label.json ./label_IMDb.json -f

echo "Downloading the RTU data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/rtu.tar.gz

echo "Untarring the RTU data..."
tar -zxvf rtu.tar.gz
mv -f rtu.tar.gz data/raw/RTU/
#rm -fr aclImdb_v1.tar.gz

echo "Converting the RTU data to json..."
python3 ../../datasets/sentiment/RTU/convert_RTU_to_JSON.py ./

cp ../../datasets/sentiment/RTU/label.json ./label_RTU.json -f

mv -f rtu/ data/raw/RTU/
mv -f data.json.gz data/json/RTU/
mv -f index.json.gz data/json/RTU/

mkdir -p data/tf/merged
echo "Generating TFRecord files..."
python ../scripts/write_tfrecords_merged.py IMDb RTU
