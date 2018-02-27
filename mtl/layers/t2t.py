# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in xrange(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4. "
                     "Shape: " + str(static_shape))
  # Add support for left padding.
  if (kwargs.get("padding").upper() == "LEFT" or
      kwargs.get("padding").upper() == "CAUSAL"):
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    name = "{}_{}".format(kwargs.get("name", "conv"), name_suffix)
    original_name = kwargs.pop("name", None)
    original_force2d = kwargs.pop("force2d", None)
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  return conv_internal(
      tf.layers.conv2d,
      inputs,
      filters,
      kernel_size,
      dilation_rate=dilation_rate,
      **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  return tf.squeeze(
      conv(
          tf.expand_dims(inputs, 2),
          filters, (kernel_size, 1),
          dilation_rate=(dilation_rate, 1),
          **kwargs), 2)
