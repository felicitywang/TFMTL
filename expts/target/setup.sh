#!/usr/bin/env bash
set -e

export LC_ALL='en_US.utf8'

mkdir -p data/raw/
mkdir -p data/json/
mkdir -p data/tf/


# Target-based sentiment
mkdir -p data/raw/Target
mkdir -p data/json/Target

echo "Downloading the Target data..."
# TODO add instructions to get the data
wget -nc http://ugrad.cs.jhu.edu/~fxwang/acl-14-short-data.zip
unzip acl-14-short-data.zip -d ./acl-14-short-data/

echo "Converting the Target data to json..."
python3 ../../datasets/sentiment/Target/convert_Target_to_JSON.py ./acl-14-short-data/

rm -fr ./acl-14-short-data/
mv -f acl-14-short-data.zip data/raw/Target/

mv -f data.json.gz data/json/Target
mv -f index.json.gz data/json/Target/

cp ../../datasets/sentiment/Target/label.json ./label_Target.json -f


unset LC_ALL
