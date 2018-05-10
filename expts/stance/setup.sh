#!/usr/bin/env bash
set -e

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

