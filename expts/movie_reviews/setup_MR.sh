#!/usr/bin/env bash
set -e

mkdir -p data/raw/MR
mkdir -p data/json/MR


echo "Downloading the MR data..."
wget -nc http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz

tar zvxf rt-polaritydata.tar.gz
mv -f rt-polaritydata.tar.gz data/raw/MR
mv -f rt-polaritydata.README.1.0.txt data/raw/MR


echo "Converting the MR data to json..."
python3 ../../datasets/sentiment/MR/convert_MR_to_JSON.py ./
mv -f data.json.gz data/json/MR/

cp ../../datasets/sentiment/MR/label.json ./label_MR.json -f

