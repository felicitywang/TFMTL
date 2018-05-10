#!/usr/bin/env bash
set -e

mkdir -p data/raw/ABSA-L
mkdir -p data/json/ABSA-L
mkdir -p data/tf/ABSA-L

echo "Downloading the ABSA-L data..."

curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtbTJnUHRIdFBULTg" > semeval2016_task5_absa_english.zip
unzip semeval2016_task5_absa_english.zip -d semeval2016-task5-absa-english

echo "Converting the ABSA-L data to json..."
python3 ../../datasets/sentiment/ABSA-L/convert_ABSA-L_to_JSON.py ./

mv -f data.json.gz data/json/ABSA-L/
mv -f index.json.gz data/json/ABSA-L/
rm semeval2016_task5_absa_english.zip -fr
rm semeval2016-task5-absa-english/ -fr

cp ../../datasets/sentiment/ABSA-L/label.json ./label_ABSA-L.json -f

