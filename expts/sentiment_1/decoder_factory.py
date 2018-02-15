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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mlvae.decoders import unigram


def build_template(name, decoder, vocab_size, kwargs=None):
  if decoder == "unigram":
    return tf.make_template('decoder_{}'.format(name), unigram,
                            vocab_size=vocab_size)
  else:
    raise ValueError("unrecognized decoder: %s" % (decoder))


def build_decoders(arch, vocab_size, args, hp=None):
  decoders = dict()
  if arch == "bow_untied":
    for ds in args.datasets:
      decoders[ds] = build_template(ds, "unigram", vocab_size)
  elif arch == "bow_tied":
    decoder = build_template("tied_decoder", "unigram", vocab_size)
    for ds in args.datasets:
      decoders[ds] = decoder
  else:
    raise NotImplementedError("custom decoder combination not supported")
  return decoders
