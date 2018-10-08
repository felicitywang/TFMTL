# ! /usr/bin/env python

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

"""Write test.tf with given test.json.gz"""
import json
import os
import sys

from docutils.io import InputError

from mtl.util.dataset import Dataset

"""Write TFRecord for the dataset to fine-tune the model with.

Usage: python write_tfrecords_finetune.py dataset_name
args_finetune_json_path finetune_json_dir vocab_dir

dataset_name: name of the dataset, the TFRecord would be saved to
data/tf/dataset_name/
args_finetune_json_path: path of the args file in which TFRecord
configurations are written
finetune_json_dir: directory of the json file of the dataset to fine tune(
where `data.json.gz` is saved)
vocab_dir: directory of the vocabulary of the data used to train the model,
the same vocabulary should be used so that the same word would have the same id

All the arguments should keep consistent with the INIT dataset, except for 
text_field_name/label_field_name/label_type/train_ratio/valid_ratio
"""


def main():
  if len(sys.argv) != 5:
    raise InputError(
      "Usage: python write_tfrecords_finetune.py dataset_name "
      "args_finetune_json_path finetune_json_dir vocab_dir")

  dataset_name = sys.argv[1]
  args_finetune_path = sys.argv[2]
  json_dir = sys.argv[3]
  vocab_dir = sys.argv[4]

  # find the used arguments
  if os.path.exists(os.path.join(os.path.abspath(vocab_dir), 'args.json')):
    args_path = os.path.join(os.path.abspath(vocab_dir), 'args.json')
  else:
    args_path = os.path.join(vocab_dir, os.listdir(vocab_dir)[0],
                             'args.json')

  with open(args_path) as file:
    args_used = json.load(file)

  with open(args_finetune_path)as file:
    args = json.load(file)

  tfrecord_dir = os.path.join("data/tf/single/", dataset_name)
  tfrecord_dir_name = \
    "min_" + str(args['min_frequency']) + \
    "_max_" + str(args['max_frequency']) + \
    "_vocab_" + str(args['max_vocab_size']) + \
    "_doc_" + str(args['max_document_length']) + \
    "_tok_" + args['tokenizer'].replace('_tokenizer', '')
  tfrecord_dir = os.path.join(tfrecord_dir, tfrecord_dir_name)

  dataset = Dataset(

    # TODO keep consistent with the training datasets?
    max_document_length=args_used['max_document_length'],
    max_vocab_size=args_used['max_vocab_size_allowed'],
    min_frequency=args_used['min_frequency'],
    max_frequency=args_used['max_frequency'],
    # padding=args_used.get('padding', args['padding']),
    # write_bow=args_used.get('write_bow', args['write_bow']),
    # write_tfidf=args_used.get('write_tfidf', args['write_tfidf']),
    # tokenizer_=args_used.get('tokenizer', args['tokenizer']),
    # preproc=args_used.get('preproc', args.get('preproc', True)),
    # vocab_all=args_used.get('vocab_all', args.get('vocab_all', False)),
    padding=args_used['padding'],
    write_bow=args_used['write_bow'],
    write_tfidf=args_used['write_tfidf'],
    tokenizer_=args_used['tokenizer_'],
    preproc=args_used['preproc'],
    vocab_all=args_used['vocab_all'],

    # may be different
    text_field_names=args['text_field_names'],
    label_field_name=args['label_field_name'],
    label_type=args.get('label_type', 'int'),

    train_ratio=args['train_ratio'],
    valid_ratio=args['train_ratio'],

    # default in finetune mode
    json_dir=json_dir,
    tfrecord_dir=tfrecord_dir,
    vocab_dir=vocab_dir,
    generate_basic_vocab=False,
    vocab_given=True,
    vocab_name='vocab_v2i.json',
    generate_tf_record=True
  )


if __name__ == '__main__':
  main()
