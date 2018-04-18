#!/usr/bin/env bash

set -e

mkdir -p data/raw/Topic5
mkdir -p data/json/Topic5

echo "Downloading the Topic-5 data..."

# Download the SemEval 2016 Task 4 Subtask C Topic-based 5-way Twitter sentiment analysis dataset
curl -L "https://drive.google.com/uc?export=download&id=1eS67x5vedrzVVk-tcyKSrumigbJKuqH-" > semeval2016_task4c_topic-based_sentiment.zip
#curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtSGpKcjQ3cnhldmc" > semeval2016_task4b_topic-based_sentiment.zip
unzip semeval2016_task4c_topic-based_sentiment.zip -d semeval2016-task4c-topic-based-sentiment
rm semeval2016_task4c_topic-based_sentiment.zip
mv -f semeval2016-task4c-topic-based-sentiment data/raw/Topic5

#echo "Converting the Topic-5 data to json..."
#python3 ../../datasets/sentiment/Topic5/convert_Topic5_to_JSON.py ./data/raw/Topic5/
#mv -f data/raw/Topic5/data.json.gz data/json/Topic5/
#mv -f data/raw/Topic5/index.json.gz data/json/Topic5/
#
#cp -f ../../datasets/sentiment/Topic5/label.json ./data/json/Topic5/label.json
#
#echo "Generating TFRecord files..."
#python ../scripts/write_tfrecords_single.py Topic5
