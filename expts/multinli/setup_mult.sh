#!/usr/bin/env bash

set -e

mkdir -p data/raw/MultiNLI
mkdir -p data/json/MultiNLI

mkdir -p data/raw/Topic5
mkdir -p data/json/Topic5

echo "Downloading the MultiNLI data..."
# Download the Multi-NLI dataset
wget http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip
unzip multinli_0.9.zip -d multinli_0.9
rm multinli_0.9.zip
mv -f multinli_0.9 data/raw/MultiNLI

echo "Converting the MultiNLI data to json..."
# Downsample MultiNLI to 10K examples (according to Isabelle Augenstein, personal correspondence)
python3 ../../datasets/nli/MultiNLI/convert_MultiNLI_to_JSON.py ./data/raw/MultiNLI/ True 10000
mv -f data/raw/MultiNLI/data.json.gz data/json/MultiNLI/
mv -f data/raw/MultiNLI/index.json.gz data/json/MultiNLI/

cp -f ../../datasets/nli/MultiNLI/label.json ./data/json/MultiNLI/label.json


echo "Downloading the Topic-5 data..."
# Download the SemEval 2016 Task 4 Subtask C Topic-based 5-way Twitter sentiment analysis dataset
curl -L "https://drive.google.com/uc?export=download&id=1eS67x5vedrzVVk-tcyKSrumigbJKuqH-" > semeval2016_task4c_topic-based_sentiment.zip
unzip semeval2016_task4c_topic-based_sentiment.zip -d semeval2016-task4c-topic-based-sentiment
rm semeval2016_task4c_topic-based_sentiment.zip
mv -f semeval2016-task4c-topic-based-sentiment data/raw/Topic5

echo "Converting the Topic-5 data to json..."
python3 ../../datasets/sentiment/Topic5/convert_Topic5_to_JSON.py ./data/raw/Topic5/
mv -f data/raw/Topic5/data.json.gz data/json/Topic5/
mv -f data/raw/Topic5/index.json.gz data/json/Topic5/

cp -f ../../datasets/sentiment/Topic5/label.json ./data/json/Topic5/label.json


echo "Generating TFRecord files..."
#python ../scripts/write_tfrecords_single.py MultiNLI
python ../scripts/write_tfrecords_merged.py MultiNLI Topic5
