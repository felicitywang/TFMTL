# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

#!/usr/bin/env bash

# Copy Gold data to current folder

# labels: data/raw/gold/labels/
# translations: data/raw/gold/translations/{oracle, one, bop}


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

cp -fr /export/a05/mahsay/domain/goldstandard/1A/speech data/raw/gold/translations/oracle/1A/
cp -fr /export/a05/mahsay/domain/goldstandard/1A/text data/raw/gold/translations/oracle/1A/
cp -fr /export/a05/mahsay/domain/goldstandard/1B/speech data/raw/gold/translations/oracle/1B/
cp -fr /export/a05/mahsay/domain/goldstandard/1B/text data/raw/gold/translations/oracle/1B/

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


# for SPO which doesn't have gold data, make empty data.json.gz
echo "[]" > data/raw/empty.json
gzip data/raw/empty.json
mkdir -p data/json/SPO_gold_one
mkdir -p data/json/SPO_gold_bop
mkdir -p data/json/SPO_gold_oracle
cp data/raw/empty.json.gz data/json/SPO_gold_oracle/data.json.gz
cp data/raw/empty.json.gz data/json/SPO_gold_one/data.json.gz
cp data/raw/empty.json.gz data/json/SPO_gold_bop/data.json.gz

