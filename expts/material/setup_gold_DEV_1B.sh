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

# 1B DEV
# /export/a05/mahsay/domain/goldstandard/1B/DEV

# gold labels
mkdir -p data/raw/gold/labels/GOV/1B/DEV
mkdir -p data/raw/gold/labels/LIF/1B/DEV

cp /export/a05/mahsay/domain/goldstandard/1B/DEV/domain_GOV.list data/raw/gold/labels/GOV/1B/DEV
cp /export/a05/mahsay/domain/goldstandard/1B/DEV/domain_LIF.list data/raw/gold/labels/LIF/1B/DEV

# one-best translations
mkdir -p data/raw/gold/translations/one/1B/DEV
cp -fr /export/a05/mahsay/domain/goldstandard/1B/DEV/* data/raw/gold/translations/one/1B/DEV

ls -1 data/raw/gold/translations/one/1B/DEV/text | wc -l   # 204
ls -1 data/raw/gold/translations/one/1B/DEV/speech | wc -l  # 119

