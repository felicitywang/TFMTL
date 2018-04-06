#!/usr/bin/env bash

set -e

mkdir -p data/raw/Topic2
mkdir -p data/json/Topic2

echo "Downloading the Topic-2 data..."
# Download the SemEval 2016 Task 4 Subtask B Topic-based Twitter sentiment analysis dataset
curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtSGpKcjQ3cnhldmc" > semeval2016_task4b_topic-based_sentiment.zip
unzip semeval2016_task4b_topic-based_sentiment.zip -d semeval2016-task4b-topic-based-sentiment
rm semeval2016_task4b_topic-based_sentiment.zip
mv -f semeval2016-task4b-topic-based-sentiment data/raw/Topic2

echo "Converting the Topic-2 data to json..."
python3 ../../datasets/sentiment/Topic2/convert_Topic2_to_JSON.py ./data/raw/Topic2/
mv -f data/raw/Topic2/data.json.gz data/json/Topic2/
mv -f data/raw/Topic2/index.json.gz data/json/Topic2/

cp -f ../../datasets/sentiment/Topic2/label.json ./data/json/Topic2/label.json
