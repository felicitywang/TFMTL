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

### 1A DEV
### /export/a05/mahsay/domain/goldstandard/1A/DEV

# gold labels
mkdir -p data/raw/gold/labels/BUS/1A/DEV
mkdir -p data/raw/gold/labels/GOV/1A/DEV
mkdir -p data/raw/gold/labels/LAW/1A/DEV
mkdir -p data/raw/gold/labels/LIF/1A/DEV
mkdir -p data/raw/gold/labels/SPO/1A/DEV

cp /export/a05/mahsay/domain/goldstandard/1A/DEV/domain_BUS.list data/raw/gold/labels/BUS/1A/DEV
cp /export/a05/mahsay/domain/goldstandard/1A/DEV/domain_GOV.list data/raw/gold/labels/GOV/1A/DEV
cp /export/a05/mahsay/domain/goldstandard/1A/DEV/domain_LAW.list data/raw/gold/labels/LAW/1A/DEV
cp /export/a05/mahsay/domain/goldstandard/1A/DEV/domain_LIF.list data/raw/gold/labels/LIF/1A/DEV
cp /export/a05/mahsay/domain/goldstandard/1A/DEV/domain_SPO.list data/raw/gold/labels/SPO/1A/DEV

# one-best translations
cp -fr /export/a05/mahsay/domain/goldstandard/1A/DEV/* data/raw/gold/translations/one/1A/DEV

ls -1 data/raw/gold/translations/one/1A/DEV/text | wc -l   # 415
ls -1 data/raw/gold/translations/one/1A/DEV/speech | wc -l # 201


