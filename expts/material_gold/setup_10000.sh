
/export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/???/data.???posmore10000.json


# gold labels

mkdir -p data/raw/GOVg/1A/
mkdir -p data/raw/LIFg/1A/
mkdir -p data/raw/BUSg/1A/
mkdir -p data/raw/LAWg/1A/
mkdir -p data/raw/GOVg/1B/
mkdir -p data/raw/LIFg/1B/
mkdir -p data/raw/HEAg/1B/
mkdir -p data/raw/MILg/1B/

cp /export/a05/mahsay/domain/goldstandard/1A/domain_GOV.list data/raw/GOVg/1A/
cp /export/a05/mahsay/domain/goldstandard/1A/domain_LIF.list data/raw/LIFg/1A/
cp /export/a05/mahsay/domain/goldstandard/1A/domain_BUS.list data/raw/BUSg/1A/
cp /export/a05/mahsay/domain/goldstandard/1A/domain_LAW.list data/raw/LAWg/1A/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_GOV.list data/raw/GOVg/1B/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_LIF.list data/raw/LIFg/1B/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_HEA.list data/raw/HEAg/1B/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_MIL.list data/raw/MILg/1B/


# gold data oracle translation
# gold data 1-best translation

mkdir -p data/json/GOVg
mkdir -p data/json/LIFg
mkdir -p data/json/BUSg
mkdir -p data/json/LAWg
mkdir -p data/json/HEAg
mkdir -p data/json/MILg

python convert_GOLD_to_JSON.py

# synthetic data
mkdir -p data/json/GOVs/
mkdir -p data/json/LIFs/
mkdir -p data/json/HEAs/
mkdir -p data/json/BUSs/
mkdir -p data/json/BUSs/
mkdir -p data/json/LAWs/
mkdir -p data/json/MILs/
mkdir -p data/json/SPOs/


# copy synthetic data
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/GOV/data.json.gz data/json/GOVs/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LIF/data.json.gz data/json/LIFs/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/BUS/data.json.gz data/json/BUSs/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LAW/data.json.gz data/json/LAWs/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/HEA/data.json.gz data/json/HEAs/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/MIL/data.json.gz data/json/MILs/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/SPO/data.json.gz data/json/SPOs/data.json.gz

cp -lfr data/json/SPOs data/json/SPO_1000


# # TODO remove keywords from synthetic data
# mkdir -p keywords/
# cp /export/a05/mahsay/domain/Concrete-Lucene/*q keywords/
# python get_keywords.py

# # TODO downsample negative examples to 100
# python subsample_negative.py

# TODO downsample synthetic 100 pos

# combine gold + synthetic data
python create_data.py

