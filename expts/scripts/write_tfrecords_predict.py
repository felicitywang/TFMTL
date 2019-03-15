#! /usr/bin/env python

# Copyright 2018 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import json
import os
import sys

from docutils.io import InputError

from mtl.util.dataset import Dataset
from mtl.util.util import make_dir, load_json


# TODO two-sequence input?

def main():
    if len(sys.argv) != 5:
        raise InputError(
            "Usage: python write_tfrecords_predict.py dataset_args_path "
            "predict_json_path predict_tf_path vocab_dir")

    dataset_args_path = sys.argv[1]
    predict_json_path = sys.argv[2]
    predict_tf_path = sys.argv[3]
    vocab_dir = sys.argv[4]

    # find the used arguments
    if os.path.exists(os.path.join(os.path.abspath(vocab_dir), 'args.json')):
        args_path = os.path.join(os.path.abspath(vocab_dir), 'args.json')
    else:
        args_path = os.path.join(
            vocab_dir, os.listdir(vocab_dir)[0], 'args.json')

    with open(args_path) as file:
        args_used = json.load(file)

    if not os.path.exists(os.path.dirname(predict_tf_path)):
        make_dir(os.path.dirname(predict_tf_path))

    # args_DATASET.json or args_merged.json which has min_freq, max_freq,
    # max_document_length etc. information, which are used to further build
    # vocabulary

    args = load_json(dataset_args_path)
    print(args)

    dataset = Dataset(

        # keep consistent with the training datasets
        max_document_length=args_used['max_document_length'],
        max_vocab_size=args_used['max_vocab_size_allowed'],
        min_frequency=args_used['min_frequency'],
        max_frequency=args_used['max_frequency'],
        # padding=args_used.get('padding', args.get('padding', False)),
        # write_bow=args_used.get('write_bow', args.get('write_bow', False)),
        # write_tfidf=args_used.get('write_tfidf', args.get('write_tfidf', False)),
        # tokenizer_=args_used.get('tokenizer', args['tokenizer']),
        # preproc=args_used.get('preproc', args.get('preproc', True)),
        # vocab_all=args_used.get('vocab_all', args.get('vocab_all', False)),

        # use new arguments
        padding=args.get('padding', args_used.get('padding', False)),
        write_bow=args.get('write_bow', args_used.get('write_bow', False)),
        write_tfidf=args.get('write_tfidf',
                             args_used.get('write_tfidf', False)),
        tokenizer_=args.get('tokenizer', args_used.get('tokenizer_',
                                                       'lower_tokenizer')),
        stemmer=args.get('stemmer', args_used.get('stemmer', 'porter_stemmer')),
        stopword=args.get('stopwords', args_used.get('stopwords', 'nltk')),
        preproc=args.get('preproc', args_used.get('preproc', True)),
        vocab_all=args.get('vocab_all', args_used.get('vocab_all', False)),

        # may be different
        text_field_names=args['text_field_names'],
        label_field_name=args['label_field_name'],
        label_type=args.get('label_type', 'int'),

        # default in predict mode
        json_dir=None,
        tfrecord_dir=None,
        vocab_dir=vocab_dir,
        generate_basic_vocab=False,
        vocab_given=True,
        vocab_name='vocab_v2i.json',
        generate_tf_record=True,
        predict_mode=True,
        predict_json_path=predict_json_path,
        predict_tf_path=predict_tf_path
    )


if __name__ == '__main__':
    main()
