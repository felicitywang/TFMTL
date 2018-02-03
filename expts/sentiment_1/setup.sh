set -e
mkdir data
mkdir data/raw
mkdir data/json

mkdir data/raw/SSTb
mkdir data/json/SSTb
mkdir data/raw/LMRD
mkdir data/json/LMRD

wget -nc https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip
mv -f trainDevTestTrees_PTB.zip data/raw/SSTb/
python3 ../../tasks/datasets/sentiment/SSTb/convert_SSTb_to_JSON.py ./
mv -f trees data/raw/SSTb/
mv -f data.json.gz data/json/SSTb/
mv -f index.json.gz data/json/SSTb/

wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
mv -f aclImdb_v1.tar.gz data/raw/LMRD/
python3 ../../tasks/datasets/sentiment/LMRD/convert_LMRD_to_JSON.py ./
mv -f aclImdb data/raw/LMRD/
mv -f data.json.gz data/json/LMRD/
mv -f index.json.gz data/json/LMRD/


python3 write_tfrecords.py