#!/usr/bin/env bash

set -e
export LC_ALL='en_US.utf8'

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

echo "Generating TFRecord files..."


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

# write TFRecord files
python ../scripts/write_tfrecords_single.py Topic2 args_Topic2_init.json
python ../scripts/write_tfrecords_single.py Topic2 args_Topic2_expand.json
python ../scripts/write_tfrecords_merged.py Topic2 Target

unset LC_ALL
