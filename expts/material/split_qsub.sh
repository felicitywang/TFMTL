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

# e.g. python get_write.py tmpwrite

python $1 > $2.sh

split -l 2 $2.sh $2

for file in $2*; do qsub -e e -l mem_free=10G,ram_free=10G -M cnfxwang@gmail.com $file; done
