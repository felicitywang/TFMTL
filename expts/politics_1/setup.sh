#!/usr/bin/env bash
set -e

mkdir -p data/raw/FGPS
mkdir -p data/json/FGPS
mkdir -p data/raw/POLT
mkdir -p data/json/POLT

echo "Downloading the FGPS data..."
wget -nc http://people.ischool.berkeley.edu/~dbamman/emnlp2015/emnlp2015_data.tar.gz

tar zxvf emnlp2015_data.tar.gz
mv -f emnlp2015_data.tar.gz data/raw/FGPS/

echo "Converting the FGPS data to json..."
python3 ../../datasets/politics/FGPS/convert_FGPS_to_JSON.py ./
mv emnlp2015_data/ data/raw/FGPS/
mv -f data.json.gz data/json/FGPS/

cp ../../datasets/politics/FGPS/label.json ./label_FGPS.json -f

echo "Downloading the POLT data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/user.politics.time.tsv.zip

echo "Untarring the POLT data..."
unzip user.politics.time.tsv.zip
mv -f user.politics.time.tsv.zip data/raw/POLT/

echo "Converting the POLT data to json..."
python3 ../../datasets/politics/POLT/convert_POLT_to_JSON.py ./

mv -f user.politics.time.tsv data/raw/POLT/
mv -f data.json.gz data/json/POLT/

cp ../../datasets/politics/POLT/label.json ./label_POLT.json -f

mkdir -p data/tf/merged
echo "Generating TFRecord files..."
python ../scripts/write_tfrecords_merged.py FGPS POLT

