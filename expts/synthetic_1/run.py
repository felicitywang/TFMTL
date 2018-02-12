#! /usr/bin/env python

# Copyright 2018 Johns Hopkins University. All Rights Reserved.
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

import argparse as ap
import numpy as np
import tensorflow as tf

import six
from six.moves import xrange

from tflm.data.preprocessing import types_and_counts
from tflm.data.preprocessing import ngram_contexts

from tflm.hparams import update_hparams_from_args


def get_args():
  p = ap.ArgumentParser()
  p.add_argument('model', choices=['simple_mlvae'])
  p.add_argument('encoder', choices=[])
  p.add_argument('decoder', choices=[])
  p.add_argument('--arch', choices=['yzx'], default='yzx')
  p.add_argument('--seed', default=42, type=int)
  p.add_argument('--learning_rate', default=0.01, type=float)
  p.add_argument('--embed-dim', default=16, type=int)
  return p.parse_args()


def get_train_op(loss):
  return None


def get_data_iter(gen, args):
  x, y = six.next(gen())
  z_type, z_shape = infer_type_and_shape(z)

  def wrapped_gen():
    g = gen()
    while True:
      x, z = six.next(g)
      yield tuple(list(feature_fn(x)) + [z])

  ds = tf.data.Dataset.from_generator(wrapped_gen, types, shapes)
  ds = ds.repeat()  # repeat indefinitely
  ds = ds.padded_batch(args.batch_size, padded_shapes=shapes)
  ds = ds.prefetch(32)
  return ds.make_one_shot_iterator()


def get_loss(batch1, batch2, vocab_size, args):



def get_scalar_loss(losses):
  return tf.reduce_mean(losses)


def minimize(loss_op, args):
  cfg = Optimizer.H()
  cfg.lr = args.learning_rate
  train_op = Optimizer(config=cfg).minimize(loss_op)
  init_op = [tf.global_variables_initializer(),
             tf.local_variables_initializer()]
  with tf.Session() as sess:
    sess.run(init_op)
    for i in xrange(args.num_gradient_steps):
      loss, _ = sess.run([loss_op, train_op])
      if i % args.report_interval == 0:
        tf.logging.info("step=%d loss=%f", i, loss)


def unpack(batch):
  x = batch[:-1]
  z = batch[-1]
  return (x, z)


def random_seq(args):
  """ Pseudo-random sequence that shouldn't be learnable (unless the model
  is cheating).

  """
  num_symbol = 2
  seq_length = 32

  def gen():
    while True:
      x = np.random.randint(num_symbol, size=(seq_length))
      z = np.zeros([1], dtype=np.float32)
      yield (x.tolist(), z)
  it = get_data_iter(gen, args)
  batch = it.get_next()
  x, z = unpack(batch)
  losses = get_losses(x, z, num_symbol, args)
  loss_op = get_scalar_loss(losses)
  final_loss = minimize(loss_op, args)
  return final_loss


def constant_seq(args):
  """ Constant sequence all models should get.

  """
  num_symbol = 2
  seq_length = 32

  def gen():
    while True:
      x = np.ones([seq_length], dtype=np.int64)
      z = np.zeros([1], dtype=np.float32)
      yield (x.tolist(), z)
  it = get_data_iter(gen, args)
  batch = it.get_next()
  x, z = unpack(batch)
  losses = get_losses(x, z, num_symbol, args)
  loss_op = get_scalar_loss(losses)
  final_loss = minimize(loss_op, args)
  return final_loss


def sine_wave(args):
  """ A sine function with a randomly sampled frequency as z.
  """
  seq_length = 64
  num_symbol = 32

  def gen():
    while True:
      slope = np.random.randint(10) + 1  # random scale
      start_x = np.random.randint(100)  # random x-point
      y = []
      for x in xrange(start_x, start_x + seq_length):
        y += [int(((np.sin(x / slope) + 1) / 2) * num_symbol)]
      assert len(y) > 0
      if args.z:
        z = np.array([slope, start_x], dtype=np.float32)
      else:
        z = np.ones([1], dtype=np.float32)
      yield (y, z)
  it = get_data_iter(gen, args)
  batch = it.get_next()
  x, z = unpack(batch)
  losses = get_losses(x, z, num_symbol, args)
  loss_op = get_scalar_loss(losses)
  final_loss = minimize(loss_op, args)
  return final_loss


def harder_sine_wave(args):
  """ A sine function with a randomly sampled frequency as z.
  """
  seq_length = 100
  num_symbol = 64

  def gen():
    while True:
      start_x = np.random.randint(100)
      scale = np.random.randint(10) + 1
      e = np.random.randint(1, 4)
      ys = []
      for x in xrange(start_x, start_x + seq_length):
        y = int(((np.power(np.sin(x / scale), e) + 1) / 2) * num_symbol)
        assert y >= 0, "less than 0"
        assert y < num_symbol, "too big"
        ys += [y]
      assert len(ys) > 0
      if args.z:
        z = np.array([start_x, scale, e], dtype=np.float32)
      else:
        z = np.ones([3], dtype=np.float32)
      yield (ys, z)

  it = get_data_iter(gen, args)
  batch = it.get_next()
  x, z = unpack(batch)
  losses = get_losses(x, z, num_symbol, args)
  loss_op = get_scalar_loss(losses)
  final_loss = minimize(loss_op, args)
  return final_loss


def var_len(args):
  """Monotonically increasing sequences of variable length with a
  special <eos> symbol (0).

  """
  max_seq_length = 10
  num_symbol = 10

  def gen():
    while True:
      length = np.random.randint(2, max_seq_length)
      x = list(xrange(1, length))
      x += [num_symbol-1]
      if args.z:
        z = np.array([length], dtype=np.float32)
      else:
        z = np.ones([1], dtype=np.float32)
      yield (x, z)

  it = get_data_iter(gen, args)
  batch = it.get_next()
  x, z = unpack(batch)
  losses = get_losses(x, z, num_symbol, args)
  if args.check and args.z:
    init_op = [tf.global_variables_initializer(),
               tf.local_variables_initializer()]
    with tf.Session() as sess:
      sess.run(init_op)
      x_v, z_v, losses_v = sess.run([x, z, losses])
      tf.logging.info("Checking returned losses.")
      for i in xrange(args.batch_size):
        l = z_v[i]
        losses_i = losses_v[i]
        i = 0
        for loss in losses_i.tolist():
          if i >= l:
            assert loss == 0, (l, losses_i)
          i += 1
    tf.logging.info("All good!")
  loss_op = get_scalar_loss(losses)
  final_loss = minimize(loss_op, args)
  return final_loss


def main(_):
  args = get_args()
  np.random.seed(args.seed)
  tf.logging.set_verbosity(tf.logging.INFO)
  args.func(args)


if __name__ == "__main__":
  tf.app.run()
