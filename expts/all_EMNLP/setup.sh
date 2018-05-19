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

mkdir -p data/raw/FNC-1
mkdir -p data/json/FNC-1
mkdir -p data/tf/FNC-1

echo "Downloading the FNC-1 data..."

# Download the Fake News Challenge datset
mkdir fakenewschallenge ; cd fakenewschallenge
wget https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_stances.csv
wget https://raw.githubusercontent.com/FakeNewsChallenge/fnc-1/master/competition_test_bodies.csv
wget https://github.com/FakeNewsChallenge/fnc-1/archive/master.zip
unzip master.zip -d . ; mv fnc-1-master/* . -f
rm -rf fnc-1-master ; rm -f master.zip
cd ..


echo "Converting the FNC-1 data to json..."
python3 ../../datasets/sentiment/FNC-1/convert_FNC-1_to_JSON.py ./

mv -f data.json.gz data/json/FNC-1/
mv -f index.json.gz data/json/FNC-1/

mv -f fakenewschallenge/ data/raw/FNC-1/

cp ../../datasets/sentiment/FNC-1/label.json ./label_FNC-1.json -f

echo "Backing up FNC-1 data..."
cp data/json/FNC-1/ -fr data/json/FNC-1.bak

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


mkdir -p data/raw/MultiNLI
mkdir -p data/json/MultiNLI

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



mkdir -p data/raw/Stance
mkdir -p data/json/Stance
mkdir -p data/tf/Stance

echo "Downloading the Stance data..."

wget http://alt.qcri.org/semeval2016/task6/data/uploads/stancedataset.zip
# wget http://alt.qcri.org/semeval2016/task6/data/uploads/semeval2016-task6-trialdata.txt
# curl -L "https://drive.google.com/uc?export=download&id=0B2Z1kbILu3YtenFDUzM5dGZEX2s" > downloaded_Donald_Trump.txt
unzip stancedataset.zip

echo "Converting the Stance data to json..."
python3 ../../datasets/sentiment/Stance/convert_Stance_to_JSON.py ./

mv -f data.json.gz data/json/Stance/
mv -f index.json.gz data/json/Stance/
rm stancedataset.zip -fr
rm StanceDataset/ __MACOSX/ -fr
# mv -f stancedataset.zip data/raw/Stance/

cp ../../datasets/sentiment/Stance/label.json ./label_Stance.json -f

mkdir -p data/raw/Topic5
mkdir -p data/json/Topic5

echo "Downloading the Topic-5 data..."

# Download the SemEval 2016 Task 4 Subtask C Topic-based 5-way Twitter sentiment analysis dataset
curl -L "https://drive.google.com/uc?export=download&id=1eS67x5vedrzVVk-tcyKSrumigbJKuqH-" > semeval2016_task4c_topic-based_sentiment.zip
#curl -L "https://drive.google.com/uc?export=download&id=0B3emjZ5O5vDtSGpKcjQ3cnhldmc" > semeval2016_task4b_topic-based_sentiment.zip
unzip semeval2016_task4c_topic-based_sentiment.zip -d semeval2016-task4c-topic-based-sentiment
rm semeval2016_task4c_topic-based_sentiment.zip
mv -f semeval2016-task4c-topic-based-sentiment data/raw/Topic5

echo "Converting the Topic-5 data to json..."
python3 ../../datasets/sentiment/Topic5/convert_Topic5_to_JSON.py ./data/raw/Topic5/
mv -f data/raw/Topic5/data.json.gz data/json/Topic5/
mv -f data/raw/Topic5/index.json.gz data/json/Topic5/

cp -f ../../datasets/sentiment/Topic5/label.json ./data/json/Topic5/label.json

