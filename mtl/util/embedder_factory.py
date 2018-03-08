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

import tensorflow as tf

from mtl.util.embed import *

def create_embedders(embed_fns, tie_embedders, vocab_size, args, embedder_kwargs):
  # embed_fns: map from dataset name to embedding function

  # map from dataset name to embedder template
  embedders = dict()

  if tie_embedders:
    # all datasets use the same embedder
    # (same embedding function, shared parameters)
    embed_fn_set = set(embed_fns.values())
    assert len(embed_fn_set) == 1, "tied embeddings must use the same embedding function"
    embed_fn = next(iter(embed_fn_set))

    # Arguments for embedder function should be the same if the embedder is tied
    assert all([embedder_kwargs[a] == embedder_kwargs[b] for a in args.datasets for b in args.datasets])
    embedder_kwargs = embedder_kwargs[args.datasets[0]]

    embedder = tf.make_template('embedder',
                                embed_fn,
                                **embedder_kwargs)
    for ds in args.datasets:
      embedders[ds] = embedder

  else:
    # all datasets use different embedders
    # embedders may be the same function but parameterized differently
    for ds in args.datasets:
      embedder = tf.make_template('embedder_{}'.format(ds),
                                  embed_fns[ds],
                                  **embedder_kwargs[ds])
      embedders[ds] = embedder

  return embedders
