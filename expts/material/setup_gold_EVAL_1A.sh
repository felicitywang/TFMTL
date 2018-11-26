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


# labels: data/raw/gold/labels/
# translations: data/raw/gold/translations/{oracle, one, bop}

### 1A EVAL
### /export/a05/mahsay/domain/goldstandard/1A/EVAL

# gold labels
mkdir -p data/raw/gold/labels/BUS/1A/EVAL
mkdir -p data/raw/gold/labels/GOV/1A/EVAL
mkdir -p data/raw/gold/labels/LAW/1A/EVAL
mkdir -p data/raw/gold/labels/LIF/1A/EVAL
mkdir -p data/raw/gold/labels/SPO/1A/EVAL

cp /export/a05/mahsay/domain/goldstandard/1A/EVAL/domain_BUS.list data/raw/gold/labels/BUS/1A/EVAL
cp /export/a05/mahsay/domain/goldstandard/1A/EVAL/domain_GOV.list data/raw/gold/labels/GOV/1A/EVAL
cp /export/a05/mahsay/domain/goldstandard/1A/EVAL/domain_LAW.list data/raw/gold/labels/LAW/1A/EVAL
cp /export/a05/mahsay/domain/goldstandard/1A/EVAL/domain_LIF.list data/raw/gold/labels/LIF/1A/EVAL
cp /export/a05/mahsay/domain/goldstandard/1A/EVAL/domain_SPO.list data/raw/gold/labels/SPO/1A/EVAL

# one-best translations
mkdir -p data/raw/gold/translations/one/1A/EVAL/
cp -fr /export/a05/mahsay/domain/goldstandard/1A/EVAL/* data/raw/gold/translations/one/1A/EVAL

ls -1 data/raw/gold/translations/one/1A/EVAL/text | wc -l   # 9123
ls -1 data/raw/gold/translations/one/1A/EVAL/speech | wc -l # 2889


