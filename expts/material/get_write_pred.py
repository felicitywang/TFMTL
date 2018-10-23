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

"""Generate shell scripts to write TFRecord files for files to predict

Example bash script:
python ../scripts/write_tfrecords_predict.py args_nopretrain.json data/pred/json/doc/1A/DEV/t6/mt-4.asr-s5/data.json.gz data/pred/tf/doc/1A/DEV/t6/mt-4.asr-s5/GOV/min_1_max_-1_vocab_-1_doc_1000/pred.tf data/tf/single/GOV_1000/min_1_max_-1_vocab_-1_doc_1000/

Usage:
    1. Change arguments in write_pred.json(or other name)
    2. Run get_write_pred.py write_pred.json to generate shell scripts
    3. Directly run the generated commands or use split_qsub.sh to qsub and run in parallel
"""

import os

from mtl.util.util import load_json

from itertools import product
import sys


def main():
    from json_minify import json_minify
    # print(json_minify(open(sys.argv[1]).read())[410:430])
    args = load_json(sys.argv[1])
    eval_dirs = args['eval_dirs']
    text_types = args['text_types']
    subdirs = args['subdirs']
    dataset_suffixes = args['dataset_suffixes']
    root_dir = args['root_dir']
    python_path = args['python_path']
    code_path = args['code_path']
    domains = args['domains']

    num = 0

    all_dirs = {}
    for eval_dir, text_type, lang in product(
            eval_dirs, text_types, subdirs):

        directory = os.path.join(text_type, lang, eval_dir)

        if directory in all_dirs:
            all_dirs[directory].append(subdirs[lang])
        else:
            all_dirs[directory] = subdirs[lang]

    for basedir, domain, dataset_suffix in product(
            all_dirs, domains, dataset_suffixes):
        dataset_path = domain + dataset_suffix
        for subdir in all_dirs[basedir]:
            num += 1
            directory = os.path.join(basedir, subdir)

            json_path = os.path.join(root_dir, 'data/pred/json', directory,
                                     'data.json.gz')
            tf_path = os.path.join(root_dir, 'data/pred/tf/', directory,
                                   dataset_path, args['args_path'], 'pred.tf')
            arch_path = os.path.join(root_dir, 'data/tf/single',
                                     dataset_path, args['args_path'])

            # if os.path.exists(tf_path):
            #     # print('{} already exists! Skipping...')
            #     continue

            command = 'cd {}\n{} {} {} {} {} {}'.format(
                root_dir,
                python_path,
                code_path,
                os.path.join(root_dir, args['args_file_path']),
                json_path,
                tf_path,
                arch_path
            )

            print(command)


if __name__ == '__main__':
    main()
