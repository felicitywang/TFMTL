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


def create_extractors(extract_fns, tie_extractors, args, extractor_kwargs):
  # extract_fns: map from dataset name to extraction function

  # map from dataset name to extractor template
  extractors = dict()

  if tie_extractors:
    # all datasets use the same extractor
    # same extraction function, shared parameters
    extract_fn_set = set(extract_fns.values())
    assert len(extract_fn_set) == 1, "tied extractors must use " \
                                     "the same extraction function"
    extract_fn = next(iter(extract_fn_set))

    # Arguments for extractor function should be
    # the same if the extractor is tied
    assert all([extractor_kwargs[a] == extractor_kwargs[b]
                for a in args.datasets for b in args.datasets])
    extractor_kwargs = extractor_kwargs[args.datasets[0]]

    extractor = tf.make_template('extractor_shared',
                                 extract_fn,
                                 **extractor_kwargs)

    for ds in args.datasets:
      extractors[ds] = extractor

  else:
    # all datasets use different extractors
    # extractors may be the same function but parameterized differently
    for ds in args.datasets:
      extractor = tf.make_template('extractor_{}'.format(ds),
                                   extract_fns[ds],
                                   **extractor_kwargs[ds])
      extractors[ds] = extractor

  return extractors
