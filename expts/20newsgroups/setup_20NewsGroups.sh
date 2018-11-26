#!/usr/bin/env bash
set -e

mkdir -p data/raw/20NewsGroups
mkdir -p data/json/20NewsGroups


echo "Downloading the 20NewsGroups data..."
wget -nc http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz

tar zvxf 20news-bydate.tar.gz
echo "Converting the 20NewsGroups data to json..."
python3 ../../datasets/topic/20NewsGroups/convert_20NewsGroups_to_JSON.py ./


mv -f 20news* data/raw/20NewsGroups

mv -f data.json.gz data/json/20NewsGroups/
mv -f index.json.gz data/json/20NewsGroups/

cp ../../datasets/topic/20NewsGroups/label.json ./label_20NewsGroups.json -f