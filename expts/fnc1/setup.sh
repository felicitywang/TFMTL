#!/usr/bin/env bash
set -e

mkdir -p data/raw/FNC-1
mkdir -p data/json/FNC-1
mkdir -p data/tf/FNC-1

echo "Downloading the FNC-1 data..."

# Download the Fake News Challenge datset
mkdir fakenewschallenge ; cd fakenewschallenge
wget https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_stances.csv
wget https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_bodies.csv
wget https://github.com/FakeNewsChallenge/fnc-1/archive/master.zip
unzip master.zip -d . ; mv fnc-1-master/* . -f
rm -rf fnc-1-master ; rm -f master.zip
cd ..


echo "Converting the FNC-1 data to json..."
python3 ../../datasets/sentiment/FNC-1/convert_FNC-1_to_JSON.py ./

mv -f data.json.gz data/json/FNC-1/
mv -f index.json.gz data/json/FNC-1/

mv -f fakenewschallenge/ data/raw/FNC-1/

cp ../../datasets/sentiment/FNC-1/label.json ./label_FNC-1.json -f

