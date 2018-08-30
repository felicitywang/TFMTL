# copy json data
mkdir -p data/json/GOV/
mkdir -p data/json/LIF/
mkdir -p data/json/HEA/
mkdir -p data/json/BUS/
mkdir -p data/json/LAW/
mkdir -p data/json/MIL/

cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/GOV/data.json.gz data/json/GOV/data.bak.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LIF/data.json.gz data/json/LIF/data.bak.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/BUS/data.json.gz data/json/BUS/data.bak.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LAW/data.json.gz data/json/LAW/data.bak.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/HEA/data.json.gz data/json/HEA/data.bak.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/MIL/data.json.gz data/json/MIL/data.bak.json.gz

# copy keywords
mkdir -p keywords/
cp /export/a05/mahsay/domain/Concrete-Lucene/*q keywords/
python get_keywords.py

# downsample negative examples to 100
python subsample_negative.py