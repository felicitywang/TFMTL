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
# =============================================================================

"""Write TFRecord Files for the single dataset.

Usage: python write_tfrecords_single.py DATASET [config file name]
If config name not given, this would automatically search for args_DATASET.json
"""

import os
import sys

from mtl.util.dataset import Dataset
from mtl.util.util import load_json


def main(argv):
  if len(argv) == 2:
    args_name = 'args_' + argv[1] + '.json'
  else:
    args_name = argv[2]

  args = load_json(args_name)

  json_dir = "data/json/" + argv[1]

  tfrecord_dir = os.path.join("data/tf/single/", argv[1])

  preproc = True
  if 'preproc' in args:
    preproc = args['preproc']

  vocab_all = False
  if 'vocab_all' in args:
    vocab_all = args['vocab_all']

  tfrecord_dir_name = \
    "min_" + str(args['min_frequency']) + \
    "_max_" + str(args['max_frequency']) + \
    "_vocab_" + str(args['max_vocab_size']) + \
    "_doc_" + str(args['max_document_length']) + \
    "_tok_" + args['tokenizer'].replace('_tokenizer', '')

  print(tfrecord_dir_name)

  if 'pretrained_file' not in args or not args[
    'pretrained_file']:
    tfrecord_dir = os.path.join(tfrecord_dir, tfrecord_dir_name)
    dataset = Dataset(json_dir=json_dir,
                      tfrecord_dir=tfrecord_dir,
                      vocab_dir=tfrecord_dir,
                      text_field_names=args['text_field_names'],
                      label_field_name=args['label_field_name'],
                      label_type=args.get('label_type', 'int'),
                      max_document_length=args['max_document_length'],
                      max_vocab_size=args['max_vocab_size'],
                      min_frequency=args['min_frequency'],
                      max_frequency=args['max_frequency'],
                      train_ratio=args['train_ratio'],
                      valid_ratio=args['valid_ratio'],
                      subsample_ratio=args['subsample_ratio'],
                      padding=args['padding'],
                      write_bow=args['write_bow'],
                      write_tfidf=args['write_tfidf'],
                      tokenizer_=args['tokenizer'],
                      generate_basic_vocab=False,
                      vocab_given=False,
                      generate_tf_record=True,
                      preproc=preproc,
                      vocab_all=vocab_all)
  else:
    vocab_path = args['pretrained_file']
    vocab_dir = os.path.dirname(vocab_path)
    vocab_name = os.path.basename(vocab_path)

    expand_vocab = False
    if 'expand_vocab' in args:
      expand_vocab = args['expand_vocab']

    if expand_vocab:
      tfrecord_dir = os.path.join(
        tfrecord_dir,
        tfrecord_dir_name + '_' +
        vocab_name[:max(vocab_name.find('.txt'),
                        vocab_name.find('.bin.gz'),
                        vocab_name.find('.vec.zip'))] +
        '_expand')
    else:
      tfrecord_dir = os.path.join(
        tfrecord_dir,
        tfrecord_dir_name + '_' +
        vocab_name[:max(vocab_name.find('.txt'),
                        vocab_name.find('.bin.gz'),
                        vocab_name.find('.vec.zip'))] +
        '_init')

    dataset = Dataset(json_dir=json_dir,
                      tfrecord_dir=tfrecord_dir,
                      vocab_given=True,
                      vocab_dir=vocab_dir,
                      vocab_name=vocab_name,
                      text_field_names=args['text_field_names'],
                      label_field_name=args['label_field_name'],
                      label_type=args.get('label_type', 'int'),
                      max_document_length=args['max_document_length'],
                      max_vocab_size=args['max_vocab_size'],
                      min_frequency=args['min_frequency'],
                      max_frequency=args['max_frequency'],
                      train_ratio=args['train_ratio'],
                      valid_ratio=args['valid_ratio'],
                      subsample_ratio=args['subsample_ratio'],
                      padding=args['padding'],
                      write_bow=args['write_bow'],
                      write_tfidf=args['write_tfidf'],
                      tokenizer_=args['tokenizer'],
                      generate_basic_vocab=False,
                      generate_tf_record=True,
                      expand_vocab=expand_vocab,
                      preproc=preproc,
                      vocab_all=vocab_all)

  with open(os.path.join(tfrecord_dir, 'vocab_size.txt'), 'w') as f:
    f.write(str(dataset.vocab_size))

  return tfrecord_dir


if __name__ == '__main__':
  main(sys.argv)
