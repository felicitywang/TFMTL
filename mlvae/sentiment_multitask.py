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
from collections import namedtuple
from collections import defaultdict

import numpy as np
import tensorflow as tf

from tflm.metrics.clustering import accuracy
from tflm.data import InputDataset
from tflm.optim import Optimizer
from tflm.nets import unigram

from sentiment_model import cnn

from vae_common import dense_layer

from mlvae import MultiLabel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging = tf.logging

TRAIN_ITER = 'train_iter'
TEST_ITER = 'test_iter'

# This defines ALL the features that ANY model possibly needs access
# to. That is, some models will only need a subset of these features.
FEATURES = {
  'targets': tf.VarLenFeature(dtype=tf.int64),
  'length': tf.FixedLenFeature([], dtype=tf.int64),
  'label': tf.FixedLenFeature([], dtype=tf.int64)
}

def parse_args():
  p = ap.ArgumentParser()
  p.add_argument('--model', type=str,
                 help='Which model to use [mlvae|mult]')
  p.add_argument('--test', action='store_true', default=False,
                 help='Use held-out test data. WARNING: DO NOT TUNE ON TEST')
  p.add_argument('--batch_size', default=128, type=int,
                 help='Size of batch.')
  # p.add_argument('--labeled_batch_size', default=128, type=int,
  #                help='Size of labeled batch.')
  # p.add_argument('--unlabeled_batch_size', default=128, type=int,
  #                help='Size of unlabeled batch.')
  p.add_argument('--eval_batch_size', default=256, type=int,
                 help='Size of evaluation batch.')
  #p.add_argument('--max_N_train', type=int,
  #               help='Size of largest training split among all datasets.')
  p.add_argument('--embed_dim', default=128, type=int,
                 help='Word embedding size')
  p.add_argument('--encode_dim', default=128, type=int,
                 help='Encoder embedding size')
  p.add_argument('--share_embed', action='store_true', default=False,
                 help='Whether datasets share word embeddings')
  p.add_argument('--share_decoders', action='store_true', default=False,
                 help='Whether decoders are shared across datasets')
  p.add_argument('--alpha', default=10.0, type=float,
                 help='Weight assigned to discriminative term in the objective')
  p.add_argument('--beta', choices=['empirical', 'even'], default='even',
                 help='How the unsupervised and supervised batches are weighted')
  p.add_argument('--lr0', default=0.0002, type=float,
                 help='Initial learning rate')
  p.add_argument('--max_grad_norm', default=5.0, type=float,
                 help='Clip gradients to max_grad_norm during training.')
  # p.add_argument('--setting', choices=['sup', 'unsup', 'semisup'],
  #                default='semisup',
  #                help='Training regime')
  #p.add_argument('--optim_style', choices=['combined', 'alternating'],
  #               default='combined',
  #               help='Semi-sup opt style (combined or alternating updates)')
  # p.add_argument('--warmup_iter', default=500, type=int,
  #                help='[semisup] Pre-train q(y | x) for this many iter.')
  p.add_argument('--supervised_loss', choices=['hybrid', 'discriminative'],
                 default='discriminative',
                 help='Loss function for supervised training.')
  p.add_argument('--gauss_kl', choices=['exact', 'approx'], default='approx',
                 help='MCMC estimate of Gaussian KL or analytic computation')
  # p.add_argument('--mlp_activation', type=check_activation, default='selu',
  #                help='Activation to use for intermediate MLP layers')
  # p.add_argument('--encoder_keep_prob', default=1.0, type=float,
  #                help='Encoder dropout probability (only used during training)')
  # p.add_argument('--latent_dim', default=64, type=int,
  #                help='Latent embedding dimensionality')
  # p.add_argument('--encoder_hidden_dim', default=512, type=int,
  #                help='Dimensionality of the hidden layers of the encoder MLP')
  # p.add_argument('--encoder_output_dim', default=512, type=int,
  #                help='Dimensionality of the output layer of the encoder MLP')
  # p.add_argument('--decoder_hidden_dim', default=512, type=int,
  #                help='Dimensionality of the hidden layers of the decoder MLP')
  p.add_argument('--num_train_epochs', default=100, type=int,
                 help='Number of training epochs.')
  # p.add_argument('--q_z_min_var', default=0.0001, type=float,
  #                help='Minimum value of the Gaussian variances')
  # p.add_argument('--percent_supervised', default=1, type=float,
  #                help=('Fraction of supervised data. If 0, an alignment',
  #                      'will be used to compute held-out accuracy'))
  p.add_argument('--print_trainable_variables', action='store_true',
                 default=False,
                 help='Diagnostic: print trainable variables')
  # p.add_argument('--num_parallel_calls', default=4, type=int,
  #                help='Number of threads used to preprocess the data')
  p.add_argument('--seed', default=42, type=int,
                 help='RNG seed')
  p.add_argument('--check_numerics', action='store_true', default=False,
                 help='Add operations to check for NaNs')
  p.add_argument('--log_device_placement', action='store_true', default=False,
                 help='Log where compute graph is placed.')
  p.add_argument('--force_gpu_compatible', action='store_true', default=False,
                 help='Throw error if any operations are not GPU-compatible')
  p.add_argument('--tau0', default=0.5, type=float,
                 help='Annealing parameter for Concrete/Gumbal-Softmax')
  p.add_argument('--label_key', default="LABELS", type=str,
                 help='Key for label field in the batches')
  p.add_argument('--datasets', nargs='+', type=str,
                 help="Key of the dataset(s) to train and evaluate on [IMDB|SSTb]")
  return p.parse_args()


def encoder_graph(inputs, embed_fn, encode_dim):
  embed = embed_fn(inputs)
  return cnn(embed,
             encode_dim=encode_dim)

def build_encoders(vocab_size, args):
  encoders = dict()
  if args.share_embed:
    # One shared word embedding matrix for all datasets
    embed_temp = tf.make_template('embedding', tf.contrib.layers.embed_sequence,
                                  vocab_size=vocab_size,
                                  embed_dim=args.embed_dim)
    for ds in args.datasets:
      encoders[ds] = tf.make_template('encoder_{}'.format(ds), encoder_graph,
                                      embed_fn=embed_temp,
                                      encode_dim=args.encode_dim)
  else:
    # A unique word embedding matrix for each dataset
    for ds in args.datasets:
      embed_temp = tf.make_template('embedding_{}'.format(ds), tf.contrib.layers.embed_sequence,
                                    vocab_size=vocab_size,
                                    embed_dim=args.embed_dim)
      encoders[ds] = tf.make_template('encoder_{}'.format(ds), encoder_graph,
                                      embed_fn=embed_temp,
                                      encode_dim=args.encode_dim)

  return encoders

def build_decoders(???):
  decoders = dict()
  if args.share_decoders:
    raise "TODO"
  else:
    raise "TODO"

  return decoders

def get_num_records(tf_record_filename):
  c = 0
  for record in tf.python_io.tf_record_iterator(tf_record_filename):
      c += 1
  return c

def train_model(model, dataset_info, steps_per_epoch, args):
  # DO we need this?
  dataset_info, model_info = fill_info_dicts(dataset_info, args)

  # Build compute graph
  logging.info("Creating computation graph.")

  # Get training objective. The inputs are:
  #   1. A dict of { dataset_key: dataset_iterator }
  #
  train_batches = {name: model_info[name]['train_batch'] for name in model_info}
  loss = model.get_multi_task_loss(train_batches)

  # Done building compute graph; set up training ops.
  
  # Training ops
  global_step_tensor = tf.train.get_or_create_global_step()
  zero_global_step_op = global_step_tensor.assign(0)
  lr = get_learning_rate(args.lr0)
  tvars, grads = get_var_grads(loss)
  train_op = get_train_op(tvars, grads, lr, args.max_grad_norm,
                          global_step_tensor, name='train_op')
  init_ops = [tf.global_variables_initializer(),
              tf.local_variables_initializer()]
  config = get_proto_config(args)
  with tf.train.SingularMonitoredSession(config=config) as sess:
    # Initialize model parameters
    sess.run(init_ops)

    # Do training
    for epoch in xrange(args.num_train_epochs):
      # Take steps_per_epoch gradient steps
      total_loss = 0
      num_iter = 0
      for _ in xrange(steps_per_epoch):
        step, loss_v, _ = sess.run([global_step_tensor, loss, train_op])
        num_iter += 1
        total_loss += loss_v  # loss_v is average loss *per training example*
      assert num_iter > 0

      # average loss per batch (which is in turn averaged across examples)
      train_loss = float(total_loss) / float(num_iter)  

      # Evaluate held-out accuracy
      if not args.test:  # Validation mode
        # Get performance metrics on each dataset
        for dataset_name in dataset_info:
          _pred_op = model_info[dataset_name]['test_pred_op']
          _eval_labels = model_info[dataset_name]['test_batch'][args.label_key]
          _eval_iterator = dataset_info[dataset_name]['test_iter']
          _metrics = compute_held_out_performance(sess, _pred_op,
                                                  _eval_labels,
                                                  _eval_iter, args)
          model_info[dataset_name]['test_metrics'] = _metrics

        # Log performance(s)
        str_ = '[epoch=%d step=%d] train_loss=%s' % (epoch, np.asscalar(step), train_loss)
        for dataset_name in model_info:
          _num_eval_total = model_info[dataset_name]['test_metrics']['ntotal']
          _eval_acc = model_info[dataset_name]['test_metrics']['accuracy']
          _eval_align_acc = model_info[dataset_name]['test_metrics']['aligned_accuracy']
          str_ += ' num_eval_total (%s)=%d eval_acc (%s)=%f eval_align_acc (%s)=%f' % (dataset_name, _num_eval_total, dataset_name, _eval_acc, dataset_name, _eval_align_acc)
      logging.info(str_)
      else:
        raise "final evaluation mode not implemented"


def compute_held_out_performance(session, pred_op, eval_label,
                                 eval_iterator, args):

  # pred_op: predicted labels
  # eval_label: gold labels

  # Initializer eval iterator
  session.run(eval_iterator.initializer)

  # Accumulate predictions
  ys = []
  y_hats = []
  while True:
    try:
      y, y_hat = session.run([eval_label, pred_op])
      assert y.shape == y_hat.shape, print(y.shape, y_hat.shape)
      ys += y.tolist()
      y_hats += y_hat.tolist()
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
    'aligned_accuracy': accuracy(ys, y_hats),
  }


def main():
  # Parse args
  args = parse_args()

  # Logging verbosity.
  logging.set_verbosity(tf.logging.INFO)

  # Seed numpy RNG
  np.random.seed(args.seed)

  dirs = dict()
  # TODO: don't hard-code these paths
  dirs['IMDB'] = "/export/b02/fwang/mlvae/tasks/datasets/sentiment/IMDB/"
  dirs['SSTb'] = "/export/b02/fwang/mlvae/tasks/datasets/sentiment/SSTb/"

  class_sizes = dict()
  class_sizes['IMDB'] = 2
  class_sizes['SSTb'] = 5

  # Ordering of concatenation for input to decoder
  ordering = dict()
  ordering['IMDB'] = 0
  ordering['SSTb'] = 1

  # Read data
  dataset_info = dict()
  for dataset_name in args.datasets:
    dataset_info[dataset_name] = dict()
    dataset_info[dataset_name]['feature_name'] = dataset_name  # feature name is just dataset name
    dataset_info[dataset_name]['dir'] = dirs[dataset_name]
    dataset_info[dataset_name]['class_size'] = class_sizes[dataset_name]
    dataset_info[dataset_name]['ordering'] = ordering[dataset_name]
    _dir = dataset_info[dataset_name]['dir']
    dataset_info[dataset_name]['dataset'] = Dataset(data_dir=_dir)

  # TODO: merge dataset vocabs (call to a function in Dataset)

  vocab_size = ??? merged_vocab.vocab_size
  
  class_sizes = {dataset_name: dataset_info[dataset_name]['class_size'] for dataset_name in dataset_info}

  order_dict = {dataset_name: dataset_info[dataset_name]['ordering'] for dataset_name in dataset_info}
  dataset_order = sorted(order_dict, key=order_dict.get)

  encoders = build_encoders(vocab_size, args)
  decoders = build_decoders()
  #decoders = {'IMDB': unigram, 'SSTb': unigram}

  # Set paths to TFRecord files
  for dataset_name in dataset_info:
    _dataset = dataset_info[dataset_name]['dataset']
    dataset_info[dataset_name]['train_path'] = _dataset.train_path
    if args.test:
      dataset_info[dataset_name]['test_path'] = _dataset.test_path
    else:
      dataset_info[dataset_name]['test_path'] = _dataset.valid_path

  # Creating the batch input pipelines.  These will load & batch
  # examples from serialized TF record files.
  for dataset_name in dataset_info:
    _train_path = dataset_info[dataset_name]['train_path']
    ds = build_input_dataset(_train_path, FEATURES, args.batch_size)
    dataset_info[dataset_name]['train_dataset'] = ds

    # Validation or test dataset
    _test_path = dataset_info[dataset_name]['test_path']
    ds = build_input_dataset(_test_path, FEATURES, args.eval_batch_size)
    dataset_info[dataset_name]['test_dataset'] = ds


  # This finds the size of the largest training dataset.
  training_files = [dataset_info[dataset_name]['train_path'] for dataset_name in dataset_info]
  max_N_train = max([get_num_records(tf_rec_file) for tf_rec_file in training_files])

  logging.info("Creating computation graph...")
  with tf.Graph().as_default():
    # Seed TensorFlow RNG
    tf.set_random_seed(args.seed)

    # Maybe check for NaNs
    if args.check_numerics:
      logging.info("Checking numerics.")
      tf.add_check_numerics_ops()

    # Maybe print out trainable variables of the model
    if args.print_trainable_variables:
      for tvar in tf.trainable_variables():
        # TODO: also print and sum up all their sizes
        print(tvar)

    # Steps per epoch.
    # One epoch: all datasets have been seen completely at least once
    steps_per_epoch = int(max_N_train / args.batch_size)  

    # Create model(s):
    # NOTE: models must support the following functions:
    #  * get_multi_task_loss()
    #        args: dictionary that maps dataset -> training batch
    #        returns: total loss accumulated over all batches in the dictionary
    #  * get_predictions()
    #        args: test batch (with <batch_len> examples), name of feature to predict
    #        returns: Tensor of size <batch_len> specifying the predicted label
    #                 for the specified feature for each of the examples in the batch
    # Seth's multi-task VAE model
    if args.model == 'mlvae':
      m = MultiLabel(class_sizes=class_sizes,
                     dataset_order=dataset_order,
                     encoders=encoders,
                     decoders=decoders,
                     hp=None)
    # Felicity's discriminative baseline
    elif args.model == 'mult':
      # TODO: Felicity: please fill in your MULT baseline here
      # m = MULT(...)
      raise "TODO"
    else:
      raise ValueError("unrecognized model: %s" % (args.model))

    # Do training
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
  return config


def fill_info_dicts(dataset_info, args):
  # Organizes inputs to the computation graph
  # The corresponding outputs are defined in train_model()

  # dataset_info: dict containing dataset-specific parameters/statistics
  #  e.g., number of classes, path to data
  # model_info: dict containing dataset-specific information w.r.t. running the model
  #  e.g., data iterators, batches, prediction operations

  # Storage for pointers to dataset-specific Tensors
  model_info = dict()
  for dataset_name in dataset_info:
    model_info[dataset_name] = dict()
  
  # Data iterators, etc.
  for dataset_name in dataset_info:
    # Training data, iterator, and batch
    _train_dataset = dataset_info[dataset_name]['train_dataset']
    _train_iter = _train_dataset.iterator
    _train_batch = _train_iter.get_next()
    model_info[dataset_name]['train_iter'] = _train_iter
    model_info[dataset_name]['train_batch'] = _train_batch

    # Held-out test data, iterator, batch, and prediction operation
    _test_dataset = dataset_info[dataset_name]['test_dataset']
    _test_iter = _test_dataset.iterator
    _test_batch = _test_iter.get_next()
    _test_pred_op = model.get_predictions(_test_batch, dataset_info[dataset_name]['feature_name'])
    model_info[dataset_name]['test_iter'] = _test_iter
    model_info[dataset_name]['test_batch'] = _test_batch
    model_info[dataset_name]['test_pred_op'] = _test_pred_op
    if args.test:
      logging.info("Using test data for evaluation.")
    else:
      logging.info("Using validation data for evaluation.")

  # Create feature_dicts for each dataset
  for dataset_name_1 in model_info:
    _feature_dict = dict()
    for dataset_name_2 in dataset_info:
      if dataset_name_1 == dataset_name_2:
        # Observe the labels (from batch)
        _feature_dict[dataset_name_1] = model_info[dataset_name_1][train_batch]
      else:
        # Don't observe the labels
        _feature_dict[dataset_name_1] = None
    model_info[dataset_name]['feature_dict'] = _feature_dict

  # Return dataset_info dict and model_info dict
  return dataset_info, model_info


def build_input_dataset(tfrecord_path, batch_size, is_training=True):
  if is_training:
    ds = InputDataset(tfrecord_path, FEATURES, batch_size,
                      num_epochs=None,  # repeat indefinitely
                      )
  else:
    ds = InputDataset(tfrecord_path, FEATURES, batch_size,
                      num_epochs=1)

  # We return the class because we might need to access the
  # initializer op for TESTING, while training only requires the
  # batches returned by the iterator.
  return ds


def get_var_grads(loss):
  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  return (tvars, grads)


if __name__ == "__main__":
  main()
                      
