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

from six.moves import xrange

import numpy as np
import tensorflow as tf

from mtl.optim import AdafactorOptimizer
from mtl.util.pipeline import Pipeline
import mtl.util.registry
from mtl.extractors import encode
from mtl.decoders import decode

VOCAB_SIZE = 8242
NUM_LABELS = 5
MAX_DOC_LEN = 57
TOKENS_FIELD = 'text'
LEN_FIELD = 'text_length'
LABEL_FIELD = 'label'
FEATURES = {
  TOKENS_FIELD: tf.VarLenFeature(dtype=tf.int64),
  LEN_FIELD: tf.FixedLenFeature([], dtype=tf.int64),
  LABEL_FIELD: tf.FixedLenFeature([], dtype=tf.int64)
}

TRAIN = 'TRAIN'
EVAL = 'EVAL'


def get_learning_rate(learning_rate):
  return tf.constant(learning_rate)


def get_train_op(tvars, grads, learning_rate, max_grad_norm,
                 optimizer, step):
  if optimizer == "adam":
    opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6,
                                 beta1=0.85, beta2=0.997)
  elif optimizer == "adafactor":
    opt = AdafactorOptimizer()
  elif optimizer == "sgd":
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == "rmsprop":
    opt = tf.train.RMSPropOptimizer(learning_rate)
  else:
    raise ValueError("unrecognized optimizer")
  if max_grad_norm > 0:
    tf.logging.info("Clipping gradients by to: %f", max_grad_norm)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
  return opt.apply_gradients(zip(grads, tvars), global_step=step)


def input_fn(file_name,
             num_epochs=None,
             shuffle=True,
             one_shot=True,
             static_max_length=None,
             num_threads=1,
             shuffle_buffer_size=10000,
             batch_size=64):
  if not tf.gfile.Exists(file_name):
    raise ValueError("%s is not valid", file_name)

  ds = Pipeline(file_name, FEATURES, batch_size=batch_size,
                num_threads=num_threads, shuffle=shuffle,
                num_epochs=num_epochs, one_shot=one_shot,
                shuffle_buffer_size=shuffle_buffer_size)

  if one_shot:
    return ds.batch
  else:
    return ds.batch, ds.iterator


def model_fn(mode, batch, hp):
  if mode == TRAIN:
    batch_size = hp.train_batch_size
  elif mode == EVAL:
    batch_size = hp.eval_batch_size
  else:
    raise ValueError("unrecognized mode: %s" % (mode))

  regularizer = None
  if embed_l2_scale > 0.0:
    regularizer = tf.contrib.layers.l2_regularizer(embed_l2_scale)
  initializer = tf.truncated_normal_initializer(mean=0.0,
                                                stddev=initializer_stddev)
  with tf.variable_scope("input_embedding", reuse=tf.AUTO_REUSE):
    embed_matrix = tf.get_variable("embed_matrix", [VOCAB_SIZE, hp.embed_dim],
                                   regularizer=regularizer,
                                   initializer=initializer)

  def embedder(x):
    x = tf.nn.embedding_lookup(embed_matrix, x)

  code = encode(batch[TOKENS_FIELD],
                batch[LEN_FIELD],
                mode == TRAIN,
                encoder=hp.encoder,


  # TODO
  #classification_error =

  avg_reconstruction_error = decode(batch[TOKENS_FIELD],
                                    batch[LEN_FIELD],
                                    VOCAB_SIZE,
                                    mode == TRAIN,
                                    embed_fn=embedder,
                                    decoder=hp.decoder,
                                    hparams=hp.decoder_hparams,
                                    average_across_timesteps=True,
                                    average_across_batch=False,
                                    global_conditioning=code)

  loss = tf.reduce_sum(losses, axis=1)
  global_step_tensor = tf.train.get_or_create_global_step()
  if mode == TRAIN:
    tvars = tf.trainable_variables()
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', loss)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if len(reg_losses) > 0:
      reg_loss = tf.reduce_sum(reg_losses)
      loss = loss + reg_loss
    grads = tf.gradients(loss, tvars)
    lr = get_learning_rate(hp.learning_rate)
    train_op = get_train_op(tvars, grads, lr, hp.max_grad_norm,
                            hp.optimizer, global_step_tensor)
    return train_op, global_step_tensor

  if mode == EVAL:
    return {
      'loss': tf.reduce_sum(loss),
      'length': tf.reduce_sum(batch[LEN_FIELD])
    }
