#!/usr/bin/env bash
set -e

mkdir -p data/raw/ABSA-R
mkdir -p data/json/ABSA-R
mkdir -p data/tf/ABSA-R

echo "Downloading the ABSA-R data..."

curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtbTJnUHRIdFBULTg" > semeval2016_task5_absa_english.zip
unzip semeval2016_task5_absa_english.zip -d semeval2016-task5-absa-english

echo "Converting the ABSA-R data to json..."
python3 ../../datasets/sentiment/ABSA-R/convert_ABSA-R_to_JSON.py ./

mv -f data.json.gz data/json/ABSA-R/
mv -f index.json.gz data/json/ABSA-R/
rm semeval2016_task5_absa_english.zip -fr
rm semeval2016-task5-absa-english/ -fr

cp ../../datasets/sentiment/ABSA-R/label.json ./label_ABSA-R.json -f

