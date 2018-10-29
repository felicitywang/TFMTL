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



# parabank-extended synthetic data
mkdir -p data/json/GOV_syn_p1000r1000_para/
mkdir -p data/json/LIF_syn_p1000r1000_para/
mkdir -p data/json/HEA_syn_p1000r1000_para/
mkdir -p data/json/BUS_syn_p1000r1000_para/
mkdir -p data/json/BUS_syn_p1000r1000_para/
mkdir -p data/json/LAW_syn_p1000r1000_para/
mkdir -p data/json/MIL_syn_p1000r1000_para/
mkdir -p data/json/SPO_syn_p1000r1000_para/


cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/GOV/data.json.gz data/json/GOV_syn_p1000r1000_para/data.json.gz
cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/LIF/data.json.gz data/json/LIF_syn_p1000r1000_para/data.json.gz
cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/BUS/data.json.gz data/json/BUS_syn_p1000r1000_para/data.json.gz
cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/LAW/data.json.gz data/json/LAW_syn_p1000r1000_para/data.json.gz
cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/HEA/data.json.gz data/json/HEA_syn_p1000r1000_para/data.json.gz
cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/MIL/data.json.gz data/json/MIL_syn_p1000r1000_para/data.json.gz
cp /export/a05/mahsay/domain/data/json/syn/p1000r1000-para/SPO/data.json.gz data/json/SPO_syn_p1000r1000_para/data.json.gz



# parabank-extended turk data

cp -fr /export/a05/mahsay/domain/data/json/turk/para/ data/raw

for file in data/raw/para/* ; do mv "$file" "$file"_para ;  done

mv data/raw/para/* data/json


