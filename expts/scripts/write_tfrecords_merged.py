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

import os
import sys

from mtl.util.dataset import merge_dict_write_tfrecord, \
  merge_pretrain_write_tfrecord
from mtl.util.util import make_dir, load_json


def main(argv):
  if argv[-1].endswith('.json'):
    args_name = argv[-1]
    argv = argv[:-1]
  else:
    args_name = 'args_merged.json'
  args = load_json(args_name)

  tfrecord_dir = "data/tf/merged/"
  datasets = sorted(argv[1:])
  for dataset in datasets[:-1]:
    tfrecord_dir += dataset + "_"
  tfrecord_dir += datasets[-1] + '/'

  json_dirs = [os.path.join('data/json/', dataset) for dataset in datasets]

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

  if 'pretrained_file' not in args or not args[
    'pretrained_file']:
    tfrecord_dir = os.path.join(tfrecord_dir, tfrecord_dir_name)
    tfrecord_dirs = [os.path.join(tfrecord_dir, dataset) for dataset in
                     datasets]
    assert [os.path.basename(tf_dir) for tf_dir in tfrecord_dirs] == [
      os.path.basename(json_dir) for json_dir in json_dirs]
    for i in tfrecord_dirs:
      make_dir(i)
    merge_dict_write_tfrecord(json_dirs=json_dirs,
                              tfrecord_dirs=tfrecord_dirs,
                              merged_dir=tfrecord_dir,
                              max_document_length=args[
                                'max_document_length'],
                              max_vocab_size=args['max_vocab_size'],
                              min_frequency=args['min_frequency'],
                              max_frequency=args['max_frequency'],
                              text_field_names=args['text_field_names'],
                              label_field_name=args['label_field_name'],
                              train_ratio=args['train_ratio'],
                              valid_ratio=args['valid_ratio'],
                              tokenizer_=args['tokenizer'],
                              subsample_ratio=args['subsample_ratio'],
                              padding=args['padding'],
                              write_bow=args['write_bow'],
                              write_tfidf=args['write_tfidf'],
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

    tfrecord_dirs = [os.path.join(tfrecord_dir, dataset) for dataset in
                     datasets]
    for i in tfrecord_dirs:
      make_dir(i)
    assert [os.path.basename(tf_dir) for tf_dir in tfrecord_dirs] == [
      os.path.basename(json_dir) for json_dir in json_dirs]

    merge_pretrain_write_tfrecord(json_dirs=json_dirs,
                                  tfrecord_dirs=tfrecord_dirs,
                                  merged_dir=tfrecord_dir,
                                  vocab_dir=vocab_dir,
                                  vocab_name=vocab_name,
                                  text_field_names=args[
                                    'text_field_names'],
                                  label_field_name=args[
                                    'label_field_name'],
                                  max_document_length=args[
                                    'max_document_length'],
                                  max_vocab_size=args['max_vocab_size'],
                                  min_frequency=args['min_frequency'],
                                  max_frequency=args['max_frequency'],
                                  train_ratio=args['train_ratio'],
                                  valid_ratio=args['valid_ratio'],
                                  subsample_ratio=args[
                                    'subsample_ratio'],
                                  padding=args['padding'],
                                  write_bow=args['write_bow'],
                                  write_tfidf=args['write_tfidf'],
                                  tokenizer_=args['tokenizer'],
                                  expand_vocab=expand_vocab,
                                  preproc=preproc,
                                  vocab_all=vocab_all)

  return tfrecord_dir


if __name__ == '__main__':
  main(sys.argv)
