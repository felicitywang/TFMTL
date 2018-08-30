#!/usr/bin/env bash

set -e
export LC_ALL='en_US.utf8'

# two text input

file="data/json/Topic2/data.json.gz"
if [ -f "$file" ]
then
    echo "$file already exists."
else
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
fi

file="data/json/Target/data.json.gz"
if [ -f "$file" ]
then
    echo "$file already exists."
else
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
fi

# write TFRecord files
python ../scripts/write_tfrecords_single.py Topic2 args_Topic2_init_glove.json
python ../scripts/write_tfrecords_single.py Topic2 args_Topic2_expand_glove.json
python ../scripts/write_tfrecords_merged.py Target Topic2

# one text input

file="data/json/SSTb/data.json.gz"
if [ -f "$file" ]
then
	echo "$file already exists."
else
    mkdir -p data/raw/SSTb
    mkdir -p data/json/SSTb

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

fi

file="data/json/LMRD/data.json.gz"
if [ -f "$file" ]
then
    echo "$file already exists."
else
    mkdir -p data/raw/LMRD
    mkdir -p data/json/LMRD

    echo "Downloading the LMRD data..."
    wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

    echo "Untarring the LMRD data..."
    tar zxvf aclImdb_v1.tar.gz
    mv -f aclImdb_v1.tar.gz data/raw/LMRD/
    #rm -fr aclImdb_v1.tar.gz

    echo "Converting the LMRD data to json..."
    python3 ../../datasets/sentiment/LMRD/convert_LMRD_to_JSON.py ./

    mv -f aclImdb data/raw/LMRD/
    mv -f data.json.gz data/json/LMRD/
    mv -f index.json.gz data/json/LMRD/

    mkdir -p data/tf/merged
    echo "Generating TFRecord files..."
    python ../scripts/write_tfrecords_merged.py LMRD SSTb
fi

# write TFRecord files
python ../scripts/write_tfrecords_single.py SSTb args_SSTb_init_glove.json
# python ../scripts/write_tfrecords_single.py SSTb args_Topic2_expand_glove.json
python ../scripts/write_tfrecords_merged.py SSTb LMRD args_merged_single_input.json

unset LC_ALL
