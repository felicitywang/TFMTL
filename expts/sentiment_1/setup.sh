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
mv trainDevTestTrees_PTB.zip data/raw/SSTb/
python3 ../../tasks/datasets/sentiment/SSTb/convert_SSTb_to_JSON.py ./
mv trees data/raw/SSTb/ -f
mv data.json.gz data/json/SSTb/ -f
mv index.json.gz data/json/SSTb/ -f

wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
mv aclImdb_v1.tar.gz data/raw/LMRD/ -f
python3 ../../tasks/datasets/sentiment/LMRD/convert_LMRD_to_JSON.py ./
mv aclImdb data/raw/LMRD/ -fr
mv data.json.gz data/json/LMRD/ -f
mv index.json.gz data/json/LMRD/ -f


python3 write_tfrecords.py