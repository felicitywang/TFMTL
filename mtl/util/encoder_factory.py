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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from mtl.util.embedder_factory import create_embedders
from mtl.util.extractor_factory import create_extractors

from mtl.util.embed import *

from mtl.encoders.paragram import paragram_phrase
from mtl.encoders.cnn import conv_and_pool
from mtl.encoders.rnn import rnn_and_pool
from mtl.encoders.no_op import no_op_encoding

from mtl.util.reducers import *

FUNCTIONS = {
  "embed_sequence": embed_sequence,
  "no_op_embedding": no_op_embedding,

  "paragram": paragram_phrase,
  "conv_and_pool": conv_and_pool,
  "rnn_and_pool": rnn_and_pool,
  "no_op_encoding": no_op_encoding,

  "reduce_min_over_time": reduce_min_over_time,
  "reduce_max_over_time": reduce_max_over_time,
  "reduce_avg_over_time": reduce_avg_over_time,
  "reduce_var_over_time": reduce_var_over_time,

  "tf.nn.relu": tf.nn.relu,
  "tf.nn.elu": tf.nn.elu,
  }


def encoder_fn(inputs, lengths, embed_fn, extract_fn):
  e = embed_fn(inputs)
  return extract_fn(e, lengths)

def create_encoders(embedders, extractors, args):
  # combine embedder and extractor for each dataset

  # map from dataset name to encoder template
  encoders = dict()
  for ds in args.datasets:
    # TODO: does this still tie together embed/extract params?
    encoder = tf.make_template('encoder_{}'.format(ds),
                               encoder_fn,
                               embed_fn=embedders[ds],
                               extract_fn=extractors[ds])
    encoders[ds] = encoder

  return encoders

def string_to_func(d, f):
  """Converts all strings in a dictionary to their
  corresponding functions (if applicable)"""
  res = dict()
  for k, v in d.items():
    if isinstance(v, dict):
      res[k] = string_to_func(v, f)
    else:
      res[k] = f[v] if v in f else v
  return res
      
def build_encoders(vocab_size, args):
  encoders = dict()

  # Read in architectures from config file
  with open(args.encoder_config_file, 'r') as f:
    architectures = json.load(f)

  # Convert all strings in config into functions (look-up table)
  architectures = string_to_func(architectures, FUNCTIONS)

  arch = args.architecture

  embed_fns = {ds: architectures[arch][ds]['embed_fn'] for ds in architectures[arch] if type(architectures[arch][ds]) is dict}
  embed_kwargs = {ds: architectures[arch][ds]['embed_kwargs'] for ds in architectures[arch] if type(architectures[arch][ds]) is dict}
  
  extract_fns = {ds: architectures[arch][ds]['extract_fn'] for ds in architectures[arch] if type(architectures[arch][ds]) is dict}
  extract_kwargs = {ds: architectures[arch][ds]['extract_kwargs'] for ds in architectures[arch] if type(architectures[arch][ds]) is dict}

  tie_embedders = architectures[arch]['embedders_tied']
  tie_extractors = architectures[arch]['extractors_tied']
  
  embedders = create_embedders(embed_fns, tie_embedders, vocab_size=vocab_size, args=args, embedder_kwargs=embed_kwargs)
  extractors = create_extractors(extract_fns, tie_extractors, args=args, extractor_kwargs=extract_kwargs)
  encoders = create_encoders(embedders, extractors, args)

  return encoders
