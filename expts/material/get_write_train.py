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

"""Generate shell scripts to write TFRecord data

Usage:
    1. Change arguments in write_train.json(or other name)
    2. Run get_write_train.py write_train.json to generate shell scripts
    3. Directly run the generated commands or use split_qsub.sh to qsub and run in parallel
"""
import sys
from itertools import product

from mtl.util.util import load_json


def main():
    args = load_json(sys.argv[1])

    for domain, dataset_suffix, args_file_path in product(
            args['domains'], args['dataset_suffixes'], args['args_file_paths']):
        print('cd {}'.format(args['root_dir']))
        print(
            '{} {} {}{} {}'.format(
                args['python_path'],
                args['code_path'],
                domain,
                dataset_suffix,
                args_file_path
            )
        )


if __name__ == '__main__':
    main()
