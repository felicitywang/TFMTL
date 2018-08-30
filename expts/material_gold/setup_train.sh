# TODO Comment


# gold labels
mkdir -p data/raw/gold/labels/GOV/1A
mkdir -p data/raw/gold/labels/LIF/1A
mkdir -p data/raw/gold/labels/BUS/1A
mkdir -p data/raw/gold/labels/LAW/1A
mkdir -p data/raw/gold/labels/GOV/1B
mkdir -p data/raw/gold/labels/LIF/1B
mkdir -p data/raw/gold/labels/HEA/1B
mkdir -p data/raw/gold/labels/MIL/1B

cp /export/a05/mahsay/domain/goldstandard/1A/domain_GOV.list data/raw/gold/labels/GOV/1A/
cp /export/a05/mahsay/domain/goldstandard/1A/domain_LIF.list data/raw/gold/labels/LIF/1A/
cp /export/a05/mahsay/domain/goldstandard/1A/domain_BUS.list data/raw/gold/labels/BUS/1A/
cp /export/a05/mahsay/domain/goldstandard/1A/domain_LAW.list data/raw/gold/labels/LAW/1A/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_GOV.list data/raw/gold/labels/GOV/1B/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_LIF.list data/raw/gold/labels/LIF/1B/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_HEA.list data/raw/gold/labels/HEA/1B/
cp /export/a05/mahsay/domain/goldstandard/1B/domain_MIL.list data/raw/gold/labels/MIL/1B/


# gold data translations

# oracle
mkdir -p data/raw/gold/translations/oracle/1A
mkdir -p data/raw/gold/translations/oracle/1B

cp -fr /export/a05/mahsay/domain/goldstandard/1A/speech data/raw/gold/translations/oracle/1A/speech
cp -fr /export/a05/mahsay/domain/goldstandard/1A/text data/raw/gold/translations/oracle/1A/text
cp -fr /export/a05/mahsay/domain/goldstandard/1B/speech data/raw/gold/translations/oracle/1B/speech
cp -fr /export/a05/mahsay/domain/goldstandard/1B/text data/raw/gold/translations/oracle/1B/text

# one-best
mkdir -p data/raw/gold/translations/one/1A/speech
mkdir -p data/raw/gold/translations/one/1A/text
mkdir -p data/raw/gold/translations/one/1B/speech
mkdir -p data/raw/gold/translations/one/1B/text

cp -fr /export/a05/mahsay/MATERIAL/1A/goldDOMAIN/t6/mt-4.asr-s5/* data/raw/gold/translations/one/1A/speech
cp -fr /export/a05/mahsay/MATERIAL/1A/goldDOMAIN/tt18/* data/raw/gold/translations/one/1A/text
cp -fr /export/a05/mahsay/MATERIAL/1B/goldDOMAIN/t6/mt-5.asr-s5/* data/raw/gold/translations/one/1B/speech
cp -fr /export/a05/mahsay/MATERIAL/1B/goldDOMAIN/tt20/* data/raw/gold/translations/one/1B/text


# bag-of-phrase based
mkdir -p data/raw/gold/translations/bop/1A/speech
mkdir -p data/raw/gold/translations/bop/1A/text
mkdir -p data/raw/gold/translations/bop/1B/speech
mkdir -p data/raw/gold/translations/bop/1B/text

cp -fr /export/a05/mahsay/MATERIAL/1A/goldDOMAIN/t6.bop/concat/*  data/raw/gold/translations/bop/1A/speech
cp -fr /export/a05/mahsay/MATERIAL/1A/goldDOMAIN/tt18.bop/concat/* data/raw/gold/translations/bop/1A/text
cp -fr /export/a05/mahsay/MATERIAL/1B/goldDOMAIN/t6.bop/concat/* data/raw/gold/translations/bop/1B/speech
cp -fr /export/a05/mahsay/MATERIAL/1B/goldDOMAIN/tt20.bop/concat/* data/raw/gold/translations/bop/1B/text


ls -1 data/raw/gold/translations/oracle/1B/text | wc -l
ls -1 data/raw/gold/translations/one/1B/text | wc -l
ls -1 data/raw/gold/translations/bop/1B/text | wc -l


# convert to json
python convert_GOLD_to_JSON.py

# synthetic data
mkdir -p data/json/GOVs_1000/
mkdir -p data/json/LIFs_1000/
mkdir -p data/json/HEAs_1000/
mkdir -p data/json/BUSs_1000/
mkdir -p data/json/BUSs_1000/
mkdir -p data/json/LAWs_1000/
mkdir -p data/json/MILs_1000/
mkdir -p data/json/SPOs_1000/

# copy synthetic data
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/GOV/data.json.gz data/json/GOVs_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LIF/data.json.gz data/json/LIFs_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/BUS/data.json.gz data/json/BUSs_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LAW/data.json.gz data/json/LAWs_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/HEA/data.json.gz data/json/HEAs_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/MIL/data.json.gz data/json/MILs_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/SPO/data.json.gz data/json/SPOs_1000/data.json.gz


# combine gold + synthetic data
python create_data.py

