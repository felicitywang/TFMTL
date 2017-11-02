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

from collections import OrderedDict
from operator import itemgetter
from six.moves import xrange


class SymbolTable(object):
  """A bijective mapping from strings to indices. Enforces that the
  indices are a contiguous range and are sorted in order of decreasing
  frequency.

  """
  def __init__(self):
    self._freq = OrderedDict()
    self._total_count = 0
    self._frozen = False
    self._unk = None

  def add_n(self, x, n):
    for _ in xrange(n):
      self.add(x)

  def add(self, x):
    """Add an observation to the table.
    Args:
      x: the observed value
    """
    if type(x) == list:
      raise ValueError('call add_many() to observe multiple values')

    # Update counts
    if x in self._freq:
      self._freq[x] += 1
    else:
      self._freq[x] = 1
    self._total_count += 1

  def _get_sorted_freq(self, unk, min_freq, max_size):
    """Helper method to frequency-sort the observed types."""
    if unk:
      self._unk = unk
      unk_freq = 0

    if not min_freq:
      sorted_freq = sorted(self._freq.items(), reverse=True, key=itemgetter(1))
    else:
      unk_freq = sum([c for v, c in self._freq.items() if c < min_freq])
      sorted_freq = sorted([(v, c) for v, c in self._freq.items()
                            if c >= min_freq], reverse=True,
                           key = itemgetter(1))
      new_total = sum([c for v, c in sorted_freq])
      assert new_total + unk_freq == self.total_count, "{} != {}".format(
        new_total+unk_freq, self.total_count)

    if max_size:
      if max_size < len(sorted_freq):
        remainder = sorted_freq[max_size:]
        unk_freq = 0
        if unk:
          unk_freq += sum(c for v, c in remainder)
        sorted_freq = sorted_freq[:max_size]
        new_total = sum([c for v, c in sorted_freq])

        if unk:
          # If we have no unk symbol, then we're dropping words so the
          # total word count will be lower. If we have unk, we want to
          # make sure the total count didn't change after mapping low
          # freq words to unk.
          assert new_total + unk_freq == self.total_count, "{} != {}".format(
            new_total+unk_freq, self.total_count)

    # If we have an unk symbol, add it and sort one last time
    if unk:
      sorted_freq = sorted(sorted_freq + [(unk, unk_freq)], reverse=True,
                           key=itemgetter(1))
    return sorted_freq

  def freeze(self, unk=None, min_freq=None, max_size=None):
    """Finalize the symbol table after observing tokens. min_freq and/or
      max_size may be provided to limit the size of the resulting table.

    Args:

      unk: either a dict or value for unknown tokens. 

      min_freq: any typewith fewer than min_freq observations gets
        mapped to the corresponding unk symbol.

      max_size: if the table is larger than max_size entries, the
        least frequent types will be truncated so that it has size
        max_size.

    """

    if self.frozen:
      raise ValueError('freeze() should only be called once')

    if unk in self._freq:
      raise ValueError(
        'unk value {} already added; pick something else'.format(
          unk))

    sorted_freq = self._get_sorted_freq(unk, min_freq, max_size)

    self._i2v = dict(enumerate([v for v, c in sorted_freq]))
    self._v2i = {v: i for i, v in self._i2v.items()}
    self._frozen = True

  def idx(self, v):
    if type(v) == list:
      raise ValueError('doesnt work on lists')
    if not self.frozen:
      raise ValueError('must call freeze() first')
    if v in self._v2i:
      return self._v2i[v]
    elif self._unk:
      return self._v2i[self._unk]
    else:
      raise ValueError('{} not in symbol table'.format(v))

  def val(self, i):
    if not self.frozen:
      raise ValueError('must call freeze() first')
    if type(i) == list:
      return [self.val(x) for x in i]
    return self._i2v[i]

  def __len__(self):
    if not self.frozen:
      raise ValueError('must call freeze() first')
    return len(self._v2i)

  def __str__(self):
    ret = ""
    for k, v in self._i2v.items():
      ret += "{} <-> {}\n".format(k, v)
    ret += "\n"
    return ret

  def has_val(self, v):
    return v in self._v2i

  def has_key(self, i):
    return i in self._i2v

  @property
  def value_to_index(self):
    return self._v2i

  @property
  def index_to_value(self):
    return self._i2v

  @property
  def total_count(self):
    return self._total_count

  @property
  def frozen(self):
    return self._frozen
