# Modified from official implementation by the author from
# https://github.com/kentonl/ran


import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear as linear
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import RNNCell


class RANCell(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

    def __init__(self, num_units, input_size=None, activation=tanh,
                 normalize=False, reuse=None):
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._normalize = normalize
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                value = tf.nn.sigmoid(
                    linear([state, inputs], 2 * self._num_units, True,
                           # normalize=self._normalize
                           )
                )
                i, f = array_ops.split(value=value, num_or_size_splits=2, axis=1)

            with vs.variable_scope("candidate"):
                c = linear([inputs], self._num_units, True,
                           # normalize=self._normalize
                           )

            new_c = i * c + f * state
            # new_h = self._activation(c)
            new_h = c

        return new_h, new_c

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import tensorflow as tf
#
#
# class RANCell(tf.contrib.rnn.RNNCell):
#   def __init__(self, num_units, use_tanh=True, num_proj=None):
#     self.num_units = num_units
#     self.use_tanh = use_tanh
#     self.num_proj = num_proj
#
#   @property
#   def output_size(self):
#     if self.num_proj is not None:
#       return self.num_proj
#     else:
#       return self.num_units
#
#   @property
#   def state_size(self):
#     return tf.contrib.rnn.LSTMStateTuple(self.num_units, self.output_size)
#
#   def linear(self, inputs, output_size, use_bias=True):
#     w = tf.get_variable("w", [inputs.get_shape()[-1].value, output_size])
#     if use_bias:
#       b = tf.get_variable("b", [output_size],
#                           initializer=tf.constant_initializer())
#       return tf.nn.xw_plus_b(inputs, w, b)
#     else:
#       return tf.matmul(inputs, w)
#
#   def __call__(self, inputs, state, scope=None):
#     with tf.variable_scope(scope or type(self).__name__):
#       c, h = state
#       with tf.variable_scope("content"):
#         content = self.linear(inputs, self.num_units, use_bias=False)
#       with tf.variable_scope("gates"):
#         gates = tf.nn.sigmoid(
#           self.linear(tf.concat([inputs, h], 1), 2 * self.num_units))
#
#       i, f = tf.split(gates, num_or_size_splits=2, axis=1)
#       new_c = i * content + f * c
#       new_h = new_c
#       if self.use_tanh:
#         new_h = tf.tanh(new_h)
#       if self.num_proj is not None:
#         new_h = self.linear(new_h, self.num_proj)
#       output = new_h
#       new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
#       return output, new_state
