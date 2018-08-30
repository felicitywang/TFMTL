# synthetic data
mkdir -p data/json/GOV_syn_1000/
mkdir -p data/json/LIF_syn_1000/
mkdir -p data/json/HEA_syn_1000/
mkdir -p data/json/BUS_syn_1000/
mkdir -p data/json/BUS_syn_1000/
mkdir -p data/json/LAW_syn_1000/
mkdir -p data/json/MIL_syn_1000/
mkdir -p data/json/SPO_syn_1000/


# copy synthetic data
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/GOV/data.json.gz data/json/GOV_syn_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LIF/data.json.gz data/json/LIF_syn_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/BUS/data.json.gz data/json/BUS_syn_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/LAW/data.json.gz data/json/LAW_syn_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/HEA/data.json.gz data/json/HEA_syn_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/MIL/data.json.gz data/json/MIL_syn_1000/data.json.gz
cp /export/a05/mahsay/domain/miniscale/tfmtl/expts/glove_test/data/json/SPO/data.json.gz data/json/SPO_syn_1000/data.json.gz



