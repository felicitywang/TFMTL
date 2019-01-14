#!/usr/bin/env bash
set -e

mkdir -p data/raw/RTU
mkdir -p data/json/RTU

echo "Downloading the RTU data..."
wget -nc http://ugrad.cs.jhu.edu/~fxwang/rtu.tar.gz

echo "Untarring the RTU data..."
tar -zxvf rtu.tar.gz
mv -f rtu.tar.gz data/raw/RTU/


echo "Converting the RTU data to json..."
python3 ../../datasets/sentiment/RTU/convert_RTU_to_JSON.py ./

cp ../../datasets/sentiment/RTU/label.json ./label_RTU.json -f

mv -f RTU/ data/raw/RTU/
mv -f data.json.gz data/json/RTU/
mv -f index.json.gz data/json/RTU/

