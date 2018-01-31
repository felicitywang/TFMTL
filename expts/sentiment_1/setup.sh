mkdir data
mkdir data/raw
mkdir data/json

mkdir data/raw/SSTb
mkdir data/json/SSTb
mkdir data/raw/LMRD
mkdir data/json/LMRD

wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip
mv trainDevTestTrees_PTB.zip data/raw/SSTb/
python3 convert_json_SSTb.py
mv trees data/raw/SSTb/
mv data.json.gz data/json/SSTb/
mv index.json.gz data/json/SSTb/

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
mv aclImdb_v1.tar.gz data/raw/LMRD/
python3 convert_json_LMRD.py
mv aclImdb data/raw/LMRD/
mv data.json.gz data/json/LMRD/
mv index.json.gz data/json/LMRD/


python3 write_tfrecords.py