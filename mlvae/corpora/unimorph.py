# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
try:
  import ujson as json
except ImportError:
  import json

from functools import reduce
from tflm.data import Dataset
from tflm.data import SymbolTable


class UniMorph(Dataset):
  SOURCE_URL = None
  DEST_ARCHIVE = None
  NAME = "um"
  JSON_FILES = ("train", "valid", "test")

  def __init__(self,
               work_directory,
               language_code,
               schema_file_path,
               inverse_schema_file_path):
    self.language_code = language_code
    self.schema_file_path = schema_file_path
    self.inverse_schema_file_path = inverse_schema_file_path
    Dataset.__init__(self, work_directory)

  def json_ready(self):
    return all(os.path.exists(self._json_path(split))
               for split in self.__class__.JSON_FILES)
    
  def setup(self):
    assert self.raw_ready()

    self.load_schema(self.schema_file_path, self.inverse_schema_file_path)

    self.build_vocab()

    if not self.json_ready():
      self.json_write()    # this is not safe for parallel jobs

  def load_schema(self, schema_file_path, inverse_schema_file_path):
    schema_dict = None
    schema_inverse_dict = None
    with open(schema_file_path, 'r') as f:
      schema_dict = eval(f.read())
    with open(inverse_schema_file_path, 'r') as f:
      schema_inverse_dict = eval(f.read())

    assert schema_dict is not None
    assert schema_inverse_dict is not None

    self.schema_dict = schema_dict
    self.schema_inverse_dict = schema_inverse_dict

  def raw_ready(self):
    '''
      Assert that the dataset was downloaded and extracted correctly.
    '''
    for split in ['train', 'valid', 'test']:
      if not os.path.exists(self._raw_path(split)):
        return False
    return True

  def _split_line(self, line):
    tokens = line.split()  # [inflected-lemma string, UniMorph feature string]

    # all but the last element, which is the label string
    # this allows the input word to have spaces
    input_string = tokens[:-1]

    input_chars = [list(word) for word in input_string]

    # flatten list into just a list of chars,
    # with a <space> separating each word
    input_chars = reduce(lambda a, b: a+[' ']+b, input_chars)

    labels_string = tokens[-1]
    labels = labels_string.split(';')
    return input_chars, labels

  def build_vocab(self):
    # Initialize vocabulary
    char_vocab = SymbolTable()  # vocabulary of input characters

    # create a vocabulary of labels for each dimension in the UniMorph schema
    dimension_vocabs = dict()
    for dimension in self.schema_dict:
      dimension_vocabs[dimension] = SymbolTable()

    # Want to freeze input vocabulary on a subset of the training data
    for line in self._read_raw('train'):
      input_chars, labels = self._split_line(line)
      for char in input_chars:
        char_vocab.add(char)
      char_vocab.add(UniMorph.EOS)

      for dimension in self.schema_dict:
        for label in labels:
          if dimension == self.schema_inverse_dict[label]:
            dimension_vocabs[dimension].add(label)

    for dimension in self.schema_dict:
      # signifies absence of feature in a token
      dimension_vocabs[dimension].add('<NULL>')

    char_vocab.freeze()
    for dim, voc in dimension_vocabs.items():
      voc.freeze()
    self._vocab = {'character-vocab': char_vocab,
                   'feature-vocabs': dimension_vocabs}

  def encode(self, line):
    # break up line into input word and feature labels
    input_ids = []
    feature_ids = dict()
    input_chars, labels = self._split_line(line)

    # encode inflected lemma
    input_ids.append(self.vocab['character-vocab'].idx(UniMorph.EOS))
    for char in input_chars:
      input_ids.append(self.vocab['character-vocab'].idx(char))
    input_ids.append(self.vocab['character-vocab'].idx(UniMorph.EOS))

    # encode feature labels
    # TODO account for words with multiple labels for a given feature
    # (e.g., V;V.PTCP)
    for label in labels:
      dim = self.schema_inverse_dict[label]
      feature_ids[dim] = self.vocab['feature-vocabs'][dim].idx(label)

    for dim in self.schema_dict.keys():
      if dim not in feature_ids:
        feature_ids[dim] = self.vocab['feature-vocabs'][dim].idx('<NULL>')

    # package the example: e.g., {'inflected-lemma': [0,7,2,14,...],
    #                             'features': {'TENSE': '3', 'CASE': '1', ...}}
    example = dict()
    example['inflected-lemma'] = input_ids
    example['features'] = dict()
    for dimension in self.schema_dict:
      example['features'][dimension] = feature_ids[dimension]
    return example

  def decode(self, example):
    # Do we want to join tokens and strip EOS, and de-dict?
    # Seems like we'd like to assert x == decode(encode(x))

    # since features are a set (i.e., unordered), we can only assert:
    #   x->inflected_lemma == decode(encode(x))->inflected_lemma
    #   x->features == decode(encode(x))->features
    # i.e., cannot compare feature label strings before and after
    #   encode-decode transformations because there is no
    #   enforced ordering on features

    # decode inflected lemma
    input_chars = self.vocab['character-vocab'].val(example['inflected-lemma'])
    input_chars = [char for char in input_chars if char != UniMorph.EOS]
    input_string = ''.join(input_chars)

    # decode feature dict
    feature_ids = example['features']
    labels = []
    for dim in self.schema_dict.keys():
      label = self.vocab['feature-vocabs'][dim].val(feature_ids[dim])
      labels.append(label)
    labels = [label for label in labels if label != '<NULL>']
    labels_string = ';'.join(labels)

    return '\t'.join([input_string, labels_string])

  def _raw_path(self, split):
    return os.path.join(self.work,
                        "um-raw",
                        self.language_code,
                        "um.{}.{}.txt".format(self.language_code, split))

  def _json_path(self, split):
    return os.path.join(self.work,
                        self.language_code,
                        "um.{}.{}.json".format(self.language_code, split))

  def _read_raw(self, split):
    with open(self._raw_path(split)) as f:
      for line in f:
        yield line.rstrip()

  def _write_json(self, split):
    os.makedirs(os.path.dirname(self._json_path(split)), exist_ok=True)
    examples = []
    for line in self._read_raw(split):
      example = self.encode(line)
      examples.append(example)

    #  Examples should be written all at once because
    #  'JSON is not a framed protocol, so trying to
    #  serialize multiple objects with repeated calls
    #  to json.dump() using the same file pointer
    #  will results in an invalid JSON file.'
    #  (Python 3 docs)
    with open(self._json_path(split), 'w') as f:
      json.dump(examples, f)

  def _read_json(self, split):
    print('{}: {}'.format(self._json_path(split), os.path.isfile(self._json_path(split))))
    with open(self._json_path(split)) as f:
      examples = json.load(f)
      for line in examples:
        #print(line)
        yield line        
          
  def max_len(self):
    max_len = 0
    for example in self.train():
      max_len = max(len(example['inflected-lemma']), max_len)
    for example in self.valid():
      max_len = max(len(example['inflected-lemma']), max_len)
    for example in self.test():
      max_len = max(len(example['inflected-lemma']), max_len)
    return max_len

  def json_write(self):
    # Write preprocessed data to JSON
    for split in UniMorph.JSON_FILES:
      self._write_json(split)

  def train(self):
    return self._read_json('train')

  def valid(self):
    return self._read_json('valid')

  def test(self):
    return self._read_json('test')

  @property
  def schema(self):
    return self.schema_dict

  @property
  def inverse_schema(self):
    return self.schema_inverse_dict
