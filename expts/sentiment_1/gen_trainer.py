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

import os
import argparse as ap
from six.moves import xrange
from time import time

import numpy as np
import tensorflow as tf

from mlvae.hparams import update_hparams_from_args
from mlvae.pipeline import Pipeline
from mlvae.embed import embed_sequence
from mlvae.cnn import conv_and_pool
from mlvae.decoders import unigram

from mlvae.simple_mlvae_model import default_hparams as simple_mlvae_hparams
from mlvae.simple_mlvae_model import SimpleMultiLabel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging = tf.logging

FEATURES = {
  'label': tf.FixedLenFeature([], dtype=tf.int64),
  'types': tf.VarLenFeature(dtype=tf.int64),
  'type_counts': tf.VarLenFeature(dtype=tf.int64),
  'types_lengths': tf.FixedLenFeature([], dtype=tf.int64),
  'tokens': tf.VarLenFeature(dtype=tf.int64),
  'tokens_lengths': tf.FixedLenFeature([], dtype=tf.int64),
}

IMDB_NUM_LABEL = 2
SSTb_NUM_LABEL = 5


def parse_args():
  p = ap.ArgumentParser()
  p.add_argument('--model', type=str, default='simple_mlvae',
                 choices=['simple_mlvae'], help='Model to use.')
  p.add_argument('--test', action='store_true', default=False,
                 help='Use held-out test data. WARNING: DO NOT TUNE ON TEST')
  p.add_argument('--batch_size', default=128, type=int,
                 help='Size of batch.')
  p.add_argument('--eval_batch_size', default=256, type=int,
                 help='Size of evaluation batch.')
  p.add_argument('--embed_dim', default=256, type=int,
                 help='Word embedding size')
  p.add_argument('--share_embed', action='store_true', default=False,
                 help='Whether datasets share word embeddings')
  p.add_argument('--share_decoders', action='store_true', default=False,
                 help='Whether decoders are shared across datasets')
  p.add_argument('--lr0', default=0.0001, type=float,
                 help='Initial learning rate')
  p.add_argument('--max_grad_norm', default=5.0, type=float,
                 help='Clip gradients to max_grad_norm during training.')
  p.add_argument('--num_train_epochs', default=50, type=int,
                 help='Number of training epochs.')
  p.add_argument('--print_trainable_variables', action='store_true',
                 default=False,
                 help='Diagnostic: print trainable variables')
  p.add_argument('--seed', default=42, type=int,
                 help='RNG seed')
  p.add_argument('--check_numerics', action='store_true', default=False,
                 help='Add operations to check for NaNs')
  p.add_argument('--num_intra_threads', default=0, type=int,
                 help="""Number of threads to use for intra-op
                     parallelism. If set to 0, the system will pick
                     an appropriate number.""")
  p.add_argument('--num_inter_threads', default=0, type=int,
                 help="""Number of threads to use for inter-op
                     parallelism. If set to 0, the system will pick
                     an appropriate number.""")
  p.add_argument('--log_device_placement', action='store_true', default=False,
                 help='Log where compute graph is placed.')
  p.add_argument('--force_gpu_compatible', action='store_true', default=False,
                 help='Throw error if any operations are not GPU-compatible')
  p.add_argument('--label_key', default="label", type=str,
                 help='Key for label field in the batches')
  p.add_argument('--datasets', nargs='+', choices=["SSTb", "IMDB"], type=str,
                 help='Key of the dataset(s) to train and evaluate on [IMDB|SSTb]')
  p.add_argument('--dataset_paths', nargs='+', type=str,
                 help="""Paths to the directory containing the TFRecord files (train.tf,
                 valid.tf, test.tf) for the dataset(s) given by the
                 --datasets flag (in the same order)""")
  p.add_argument('--vocab_path', type=str,
                 help='Path to the shared vocabulary for the datasets')
  return p.parse_args()


def encoder_graph(inputs, embed_fn):
  embed = embed_fn(inputs)
  return conv_and_pool(embed)


def decoder_graph(x, z, vocab_size):
  return unigram(x, z,
                 vocab_size=vocab_size)


def build_encoders(vocab_size, args):
  encoders = dict()
  if args.share_embed:
    # One shared word embedding matrix for all datasets
    embed_temp = tf.make_template('embedding', embed_sequence,
                                  vocab_size=vocab_size,
                                  embed_dim=args.embed_dim)
    for ds in args.datasets:
      encoders[ds] = tf.make_template('encoder_{}'.format(ds), encoder_graph,
                                      embed_fn=embed_temp)
  else:
    # A unique word embedding matrix for each dataset
    for ds in args.datasets:
      embed_temp = tf.make_template('embedding_{}'.format(ds), tf.contrib.layers.embed_sequence,
                                    vocab_size=vocab_size,
                                    embed_dim=args.embed_dim)
      encoders[ds] = tf.make_template('encoder_{}'.format(ds), encoder_graph,
                                      embed_fn=embed_temp)

  return encoders


def build_decoders(vocab_size, args):
  decoders = dict()
  if args.share_decoders:
    decoder = tf.make_template('decoder', decoder_graph,
                               vocab_size=vocab_size)
    for ds in args.datasets:
      decoders[ds] = decoder
  else:
    for ds in args.datasets:
      decoders[ds] = tf.make_template('decoder_{}'.format(ds), decoder_graph,
                                      vocab_size=vocab_size)

  return decoders


def get_num_records(tf_record_filename):
  c = 0
  for record in tf.python_io.tf_record_iterator(tf_record_filename):
      c += 1
  return c


def get_vocab_size(vocab_file_path):
  with open(vocab_file_path, "r") as f:
    line = f.readline().strip()
    vocab_size = int(line)
  return vocab_size


def train_model(model, dataset_info, steps_per_epoch, args):
  dataset_info, model_info = fill_info_dicts(dataset_info, model, args)

  # Get model loss
  train_batches = {name: model_info[name]['train_batch']
                   for name in model_info}
  train_encoders = {name: model_info}
  loss = model.get_multi_task_loss(train_batches)

  # Done building compute graph; set up training ops.

  # Training ops
  global_step_tensor = tf.train.get_or_create_global_step()
  lr = get_learning_rate(args.lr0)
  tvars, grads = get_var_grads(loss)
  train_op = get_train_op(tvars, grads, lr, args.max_grad_norm,
                          global_step_tensor, name='train_op')
  init_ops = [tf.global_variables_initializer(),
              tf.local_variables_initializer()]
  config = get_proto_config(args)
  with tf.train.SingularMonitoredSession(config=config) as sess:
    tf.set_random_seed(args.seed)

    # Initialize model parameters and optimizer operations
    sess.run(init_ops)

    for dataset_name in model_info:
      _train_init_op = model_info[dataset_name]['train_init_op']
      sess.run(_train_init_op)

    # Do training
    for epoch in xrange(args.num_train_epochs):
      start_time = time()

      # Take steps_per_epoch gradient steps
      total_loss = 0
      num_iter = 0
      for i in xrange(steps_per_epoch):
        if i % 10 == 0:
          logging.info("Step %d/%d" % (i+1, steps_per_epoch))
        step, loss_v, _ = sess.run([global_step_tensor, loss, train_op])
        num_iter += 1
        total_loss += loss_v
      assert num_iter > 0

      # average loss per batch (which is in turn averaged across examples)
      train_loss = float(total_loss) / float(num_iter)

      # Evaluate held-out accuracy
      if not args.test:  # Validation mode
        # Get performance metrics on each dataset
        for dataset_name in model_info:
          pred_op = model_info[dataset_name]['test_pred_op']
          eval_labels = model_info[dataset_name]['test_batch'][args.label_key]
          eval_iter = model_info[dataset_name]['test_iter']
          metrics = compute_held_out_performance(sess, pred_op,
                                                 eval_labels,
                                                 eval_iter, args)
          model_info[dataset_name]['test_metrics'] = metrics

        end_time = time()
        elapsed = end_time - start_time

        # Log performance(s)
        str_ = '[epoch=%d/%d step=%d (%d s)] train_loss=%s' % (
          epoch+1, args.num_train_epochs, np.asscalar(step), elapsed,
          train_loss)

        tot_eval_acc = 0.0
        for dataset_name in model_info:
          num_eval_total = model_info[dataset_name]['test_metrics']['ntotal']
          eval_acc = model_info[dataset_name]['test_metrics']['accuracy']
          tot_eval_acc += eval_acc
          str_ += '\n(%s) num_eval_total=%d eval_acc=%f' % (
            dataset_name, num_eval_total, eval_acc)
        str_ += ' mean_eval_acc=%f' % (tot_eval_acc / float(len(model_info)))
        logging.info(str_)
      else:
        raise "final evaluation mode not implemented"


def compute_held_out_performance(session, pred_op, eval_label,
                                 eval_iterator, args):
  # Initialize the iterator
  session.run(eval_iterator.initializer)

  # Accumulate predictions
  ys = []
  y_hats = []
  while True:
    try:
      y, y_hat = session.run([eval_label, pred_op])
      assert y.shape == y_hat.shape, print(y.shape, y_hat.shape)
      y_list = y.tolist()
      y_list = [item for sublist in y_list for item in sublist]
      y_hat_list = y_hat.tolist()
      y_hat_list = [item for sublist in y_hat_list for item in sublist]
      ys += y_list
      y_hats += y_hat_list
    except tf.errors.OutOfRangeError:
      break

  assert len(ys) == len(y_hats)

  ntotal = len(ys)
  ncorrect = 0
  for i in xrange(len(ys)):
    if ys[i] == y_hats[i]:
      ncorrect += 1
  acc = float(ncorrect) / float(ntotal)

  return {
    'ntotal': ntotal,
    'ncorrect': ncorrect,
    'accuracy': acc,
  }


def main():
  # Parse args
  args = parse_args()

  # Logging verbosity.
  logging.set_verbosity(tf.logging.INFO)

  # Seed numpy RNG
  np.random.seed(args.seed)

  # Path to each dataset
  dirs = dict()
  _dirs = zip(args.datasets, args.dataset_paths)
  for ds, path in _dirs:
    dirs[ds] = path

  # Number of label types in each dataset
  class_sizes = dict()
  class_sizes['IMDB'] = 2
  class_sizes['SSTb'] = 5

  # Read data
  dataset_info = dict()
  for dataset_name in args.datasets:
    dataset_info[dataset_name] = dict()

    # Collect dataset information/statistics
    dataset_info[dataset_name]['feature_name'] = dataset_name
    dataset_info[dataset_name]['dir'] = dirs[dataset_name]
    dataset_info[dataset_name]['class_size'] = class_sizes[dataset_name]
    _dir = dataset_info[dataset_name]['dir']

    # Set paths to TFRecord files
    _dataset_train_path = os.path.join(_dir, "train.tf")
    if args.test:
      _dataset_test_path = os.path.join(_dir, "test.tf")
    else:
      _dataset_test_path = os.path.join(_dir, "valid.tf")
    dataset_info[dataset_name]['train_path'] = _dataset_train_path
    dataset_info[dataset_name]['test_path'] = _dataset_test_path

  vocab_size = get_vocab_size(args.vocab_path)
  tf.logging.info("vocab size: %d", vocab_size)
  class_sizes = {dataset_name: dataset_info[dataset_name]['class_size']
                 for dataset_name in dataset_info}

  logging.info("Creating computation graph...")
  with tf.Graph().as_default():
    # Creating the batch input pipelines.  These will load & batch
    # examples from serialized TF record files.
    for dataset_name in dataset_info:
      _train_path = dataset_info[dataset_name]['train_path']
      ds = build_input_dataset(_train_path, FEATURES, args.batch_size,
                               is_training=True)
      dataset_info[dataset_name]['train_dataset'] = ds

      # Validation or test dataset
      _test_path = dataset_info[dataset_name]['test_path']
      ds = build_input_dataset(_test_path, FEATURES,
                               args.eval_batch_size, is_training=False)
      dataset_info[dataset_name]['test_dataset'] = ds

    # This finds the size of the largest training dataset.
    training_files = [dataset_info[dataset_name]['train_path'] for
                      dataset_name in dataset_info]
    max_N_train = max([get_num_records(tf_rec_file) for tf_rec_file in
                       training_files])

    # Representation learning
    encoders = build_encoders(vocab_size, args)

    # Regularization
    decoders = build_decoders(vocab_size, args)

    # Maybe check for NaNs
    if args.check_numerics:
      logging.info("Checking numerics.")
      tf.add_check_numerics_ops()

    # Maybe print out trainable variables of the model
    if args.print_trainable_variables:
      for tvar in tf.trainable_variables():
        # TODO: also print and sum up all their sizes
        print(tvar)

    if args.model == 'simple_mlvae':
      hp = simple_mlvae_hparams()
      update_hparams_from_args(hp, args)
      m = SimpleMultiLabel(class_sizes=class_sizes, encoders=encoders,
                           decoders=decoders, hp=hp)
    else:
      raise ValueError("unrecognized model: %s" % (args.model))

    # Do training
    print('dataset info:')
    print(dataset_info)
    steps_per_epoch = int(max_N_train / args.batch_size)
    train_model(m, dataset_info, steps_per_epoch, args)


def get_learning_rate(learning_rate):
  return tf.constant(learning_rate)


def get_train_op(tvars, grads, learning_rate, max_grad_norm, step,
                 name=None):
  with tf.name_scope(name):
    opt = tf.train.AdamOptimizer(learning_rate,
                                 epsilon=1e-6,
                                 beta1=0.85,
                                 beta2=0.997)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    return opt.apply_gradients(zip(grads, tvars), global_step=step)


def get_proto_config(args):
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.log_device_placement = args.log_device_placement
  config.gpu_options.force_gpu_compatible = args.force_gpu_compatible
  config.intra_op_parallelism_threads = args.num_intra_threads
  config.inter_op_parallelism_threads = args.num_inter_threads
  return config


def fill_info_dicts(dataset_info, model, args):
  # Storage for pointers to dataset-specific Tensors
  model_info = dict()
  for dataset_name in dataset_info:
    model_info[dataset_name] = dict()

  # Data iterators, etc.
  for dataset_name in dataset_info:
    # Training data, iterator, and batch
    _train_dataset = dataset_info[dataset_name]['train_dataset']
    _train_iter = _train_dataset.iterator
    _train_init_op = _train_dataset.init_op
    _train_batch = _train_dataset.batch
    model_info[dataset_name]['train_iter'] = _train_iter
    model_info[dataset_name]['train_init_op'] = _train_init_op
    model_info[dataset_name]['train_batch'] = _train_batch

    # Held-out test data, iterator, batch, and prediction operation
    _test_dataset = dataset_info[dataset_name]['test_dataset']
    _test_iter = _test_dataset.iterator
    _test_batch = _test_dataset.batch
    _test_pred_op = model.get_predictions(
      _test_batch, dataset_info[dataset_name]['feature_name'])
    model_info[dataset_name]['test_iter'] = _test_iter
    model_info[dataset_name]['test_batch'] = _test_batch
    model_info[dataset_name]['test_pred_op'] = _test_pred_op
    if args.test:
      logging.info("Using test data for evaluation.")
    else:
      logging.info("Using validation data for evaluation.")

  # Return dataset_info dict and model_info dict
  return dataset_info, model_info


def build_input_dataset(tfrecord_path, batch_features, batch_size,
                        is_training=True):
  if is_training:
    ds = Pipeline(tfrecord_path, batch_features, batch_size,
                  num_epochs=None)  # repeat indefinitely
  else:
    ds = Pipeline(tfrecord_path, batch_features, batch_size,
                  num_epochs=1)
  return ds


def get_var_grads(loss, args):
  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
  return (tvars, grads)


if __name__ == "__main__":
  main()
