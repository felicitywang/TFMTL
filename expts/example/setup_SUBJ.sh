#!/usr/bin/env bash
set -e

mkdir -p data/raw/SSTb
mkdir -p data/json/SSTb
mkdir -p data/raw/SUBJ
mkdir -p data/json/SUBJ

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

echo "Downloading the SUBJ data..."
wget -nc http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz

echo "Untarring the SUBJ data..."
mkdir -p rotten_imdb/
tar -zxvf rotten_imdb.tar.gz -C rotten_imdb/
mv -f rotten_imdb.tar.gz data/raw/SUBJ/
iconv -f MACCYRILLIC -t UTF-8 < rotten_imdb/quote.tok.gt9.5000 > tmp
mv -f tmp rotten_imdb/quote.tok.gt9.5000

echo "Converting the SUBJ data to json..."
python3 ../../datasets/sentiment/SUBJ/convert_SUBJ_to_JSON.py ./

mv -f rotten_imdb/ data/raw/SUBJ/
mv -f data.json.gz data/json/SUBJ/

cp ../../datasets/sentiment/SUBJ/label.json ./label_SSTb.json -f

