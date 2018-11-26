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

# /export/a05/mahsay/domain/goldstandard/1S/ANALYSIS1/

### gold DOMAIN
# gold labels
mkdir -p data/raw/gold/labels/GOV/1S/ANALYSIS1
mkdir -p data/raw/gold/labels/MIL/1S/ANALYSIS1



cp /export/a05/mahsay/domain/goldstandard/1S/ANALYSIS1/domain_GOV.list data/raw/gold/labels/GOV/1S/ANALYSIS1
cp /export/a05/mahsay/domain/goldstandard/1S/ANALYSIS1/domain_MIL.list data/raw/gold/labels/MIL/1S/ANALYSIS1


# gold data translations

# oracle
mkdir -p data/raw/gold/translations/oracle/1S/ANALYSIS1

cp -fr /export/a05/mahsay/domain/goldstandard/1S/ANALYSIS1/speech data/raw/gold/translations/oracle/1S/ANALYSIS1
cp -fr /export/a05/mahsay/domain/goldstandard/1S/ANALYSIS1/text data/raw/gold/translations/oracle/1S/ANALYSIS1


ls -1 data/raw/gold/translations/oracle/1S/ANALYSIS1/speech | wc -l # 90
ls -1 data/raw/gold/translations/oracle/1S/ANALYSIS1/text | wc -l # 290


