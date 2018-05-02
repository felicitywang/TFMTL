# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
# Copyright 2018 Johns Hopkins University.
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

"""
For decoders:
  * Register: `registry.register_decoder`
  * List: `registry.list_decoders`
  * Retrieve by name: `registry.decoder`

For hyperparameter sets:
  * Register: `registry.register_hparams`
  * List: `registry.list_hparams`
  * Retrieve by name: `registry.hparams`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

_DECODERS = {}
_HPARAMS = {}

# Camel case to snake case utils
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def _convert_camel_to_snake(name):
  s1 = _first_cap_re.sub(r"\1_\2", name)
  return _all_cap_re.sub(r"\1_\2", s1).lower()


def default_name(obj_class):
  """Convert a class name to the registry's default name for the class.
  Args:
    obj_class: the name of a class
  Returns:
    The registry's default name for the class.
  """
  return _convert_camel_to_snake(obj_class.__name__)


def register_hparams(name=None):
  """Register an HParams set. name defaults to function name snake-cased."""

  def decorator(hp_fn, registration_name=None):
    """Registers & returns hp_fn with registration_name or default name."""
    hp_name = registration_name or default_name(hp_fn)
    if hp_name in _HPARAMS:
      raise LookupError("HParams set %s already registered." % hp_name)
    _HPARAMS[hp_name] = hp_fn
    return hp_fn

  # Handle if decorator was used without parens
  if callable(name):
    hp_fn = name
    return decorator(hp_fn, registration_name=default_name(hp_fn))

  return lambda hp_fn: decorator(hp_fn, name)


def register_decoder(name=None):
  """Register a decoder. name defaults to function name snake-cased."""

  def decorator(decoder_fn, registration_name=None):
    """Register & return decoder_fn with registration_name or default name."""
    decoder_name = registration_name or default_name(decoder_fn)
    if decoder_name in _DECODERS:
      raise LookupError("Decoder %s already registered." % decoder_name)
    _DECODERS[decoder_name] = decoder_fn
    return decoder_fn

  # Handle if decorator was used without parens
  if callable(name):
    decoder_fn = name
    return decorator(decoder_fn, registration_name=default_name(decoder_fn))

  return lambda decoder_fn: decorator(decoder_fn, name)


def display_list_by_prefix(names_list, starting_spaces=0):
  """Creates a help string for names_list grouped by prefix."""
  cur_prefix, result_lines = None, []
  space = " " * starting_spaces
  for name in sorted(names_list):
    split = name.split("_", 1)
    prefix = split[0]
    if cur_prefix != prefix:
      result_lines.append(space + prefix + ":")
      cur_prefix = prefix
    result_lines.append(space + "  * " + name)
  return "\n".join(result_lines)


def hparams(name):
  if name not in _HPARAMS:
    error_msg = "HParams set %s never registered. Sets registered:\n%s"
    raise LookupError(
      error_msg % (name,
                   display_list_by_prefix(list_hparams(), starting_spaces=4)))
  return _HPARAMS[name]


def decoder(name):
  if name not in _DECODERS:
    error_msg = "Decoder %s never registered. Decoders registered:\n%s"
    raise LookupError(
      error_msg % (name,
                   display_list_by_prefix(list_decoders(), starting_spaces=4)))
  return _DECODERS[name]


def list_hparams():
  return list(_HPARAMS)


def list_decoders():
  return list(_DECODERS)
