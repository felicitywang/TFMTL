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
# See the License for the specific lang governing permissions and
# limitations under the License.
# =============================================================================

#!/bin/sh


# Usage:
# bash split_qsub.sh get_write_train.py write_train.json tmp
# bash split_qsub.sh get_write_pred.py write_pred.json tmp



python $1 $2 > $3.sh

split -l 2 $3.sh $3split

for file in $3split*; do qsub -e e -l mem_free=10G,ram_free=10G -M cnfxwang@gmail.com $file; done
