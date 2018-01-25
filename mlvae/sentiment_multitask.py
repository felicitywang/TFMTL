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

from vae_common import dense_layer

from mlvae import MultiLabel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging = tf.logging

TRAIN = 'train'
TEST = 'test'
VALID = 'valid'

def parse_args():
  p = ap.ArgumentParser()
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
  p.add_argument('--max_N_train', type=int,
                 help='Size of largest training split among all datasets.')
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
  p.add_argument('--optim_style', choices=['combined', 'alternating'],
                 default='combined',
                 help='Semi-sup opt style (combined or alternating updates)')
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
  return p.parse_args()

def cnn(inputs,
        input_size=None,
        embed_dim=128,
        num_filter=64,
        max_width=3,
        encode_dim=256,
        activation_fn=tf.nn.elu):
  # inputs: word embeddings

  if input_size is None:
    raise ValueError("Must provide input_size.")

  filter_sizes = []
  for i in xrange(2, max_width+1):
    filter_sizes.append((i + 1, num_filter))

  # Convolutional layers
  filters = []
  for width, num_filter in filter_sizes:
    conv_i = tf.layers.conv1d(
      inputs,
      num_filter,  # dimensionality of output space (num filters)
      width,  # length of the 1D convolutional window
      data_format='channels_last',  # (batch, time, embed_dim)
      strides=1,  # stride length of the convolution
      activation=tf.nn.relu,
      padding='SAME',  # zero padding (left and right)
      name='conv_{}'.format(width))

    # Max pooling
    pool_i = tf.reduce_max(conv_i, axis=1, keep_dims=False)

    # Append the filter
    filters.append(pool_i)

    # Increment filter index
    i += 1

  # Concatenate the filters
  inputs = tf.concat(filters, 1)

  # Return a dense transform
  return dense_layer(inputs, output_size=encode_dim, name='l1',
                     activation=activation_fn)

def encoder_graph(self, inputs, encode_dim, word_embed_dim, vocab_size):
  return cnn(inputs,
             input_size=vocab_size,
             embed_dim=word_embed_dim,
             encode_dim=encode_dim)

def get_num_records(tf_record_filename):
  c = 0
  for record in tf.python_io.tf_record_iterator(tf_record_filename):
      c += 1
  return c

def train_model(model, dataset_info, steps_per_epoch, args):
  # 
  fill_info_dicts(dataset_info, args)

  # Build compute graph
  # The input to this function is a dict:
  #   { dataset_key: }
  #
  loss = model.get_multi_task_loss(train_iters)

  preds = {}
  for key for dataset_info:
    preds[key] model.get_predictions(dataset_info[key][TEST])

  # Done building compute graph; set up training ops.
  
  # Training ops
  global_step_tensor = tf.train.get_or_create_global_step()
  zero_global_step_op = global_step_tensor.assign(0)
  lr = get_learning_rate(args.lr0)
  tvars, grads = get_var_grads(loss)
  lr = get_learning_rate(args.lr0)
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
        total_loss += loss_v
      assert num_iter > 0

      # average loss per batch (which is in turn averaged across examples)
      train_loss = float(total_loss) / float(num_iter)  

      # Evaluate held-out accuracy
      for dataset_name in dataset_info:
        _pred_op = model_info[dataset_name]['valid_pred_op']
        _valid_labels = model_info[dataset_name]['valid_labels']
        _valid_iterator = dataset_info[dataset_name]['valid_iter']
        _metrics = compute_held_out_performance(sess, _pred_op,
                                               _eval_labels,
                                               _eval_iter, args)
        model_info[dataset_name]['valid_metrics'] = _metrics

      str_ = '[epoch=%d step=%d] train_loss=%s' % (epoch, np.asscalar(step), train_loss)
      for dataset_name in model_info:
        _num_eval_total = model_info[dataset_name]['valid_metrics']['ntotal']
        _eval_acc = model_info[dataset_name]['valid_metrics']['accuracy']
        _eval_align_acc = model_info[dataset_name]['valid_metrics']['aligned_accuracy']
        str_ += ' num_eval_total (%s)=%d eval_acc (%s)=%f eval_align_acc (%s)=%f' % (dataset_name, _num_eval_total, dataset_name, _eval_acc, dataset_name, _eval_align_acc)
      logging.info(str_)

    # Final test data evaluation
    for dataset_name in dataset_info:
      _pred_op = model_info[dataset_name]['test_pred_op']
      _valid_labels = model_info[dataset_name]['test_labels']
      _valid_iterator = dataset_info[dataset_name]['test_iter']
      _metrics = compute_held_out_performance(sess, _pred_op,
                                             _eval_labels,
                                             _eval_iter, args)
      model_info[dataset_name]['test_metrics'] = _metrics

    str_ = 'FINAL TEST EVAL:'
    for dataset_name in model_info:
      _num_eval_total = model_info[dataset_name]['test_metrics']['ntotal']
      _eval_acc = model_info[dataset_name]['test_metrics']['accuracy']
      _eval_align_acc = model_info[dataset_name]['test_metrics']['aligned_accuracy']
      str_ += ' num_eval_total (%s)=%d eval_acc (%s)=%f eval_align_acc (%s)=%f' % (dataset_name, _num_eval_total, dataset_name, _eval_acc, dataset_name, _eval_align_acc)
    logging.info(str_)


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

  # Read data
  dataset_info = dict()

  dataset_info['IMDB'] = dict()
  dataset_info['SSTb'] = dict()

  for dataset_name in dataset_info:
    dataset_info[dataset_name]['feature_name'] = dataset_name  # feature name is just dataset name

  dataset_info['IMDB']['dir'] = "/export/b02/fwang/mlvae/tasks/datasets/sentiment/IMDB/"
  dataset_info['SSTb']['dir'] = "/export/b02/fwang/mlvae/tasks/datasets/sentiment/SSTb/"

  dataset_info['IMDB']['class_size'] = 2
  dataset_info['SSTb']['class_size'] = 5
  class_sizes = {dataset_name: dataset_info[dataset_name]['class_size'] for dataset_name in dataset_info}

  dataset_info['IMDB']['ordering'] = 0
  dataset_info['SSTb']['ordering'] = 1
  order_dict = {dataset_name: dataset_info[dataset_name]['ordering'] for dataset_name in dataset_info}
  dataset_order = sorted(order_dict, key=order_dict.get)

  encoders = {'IMDB': ???, 'SSTb': ???}
  decoders = {'IMDB': unigram, 'SSTb': unigram}

  for dataset_name in dataset_info:
    _dir = dataset_info[dataset_name]['dir']
    dataset_info[dataset_name]['dataset'] = Dataset(data_dir=_dir)

  # num_classes = dataset.num_classes
  # max_document_length = dataset.max_document_length
  # vocab_size = dataset.vocab_size

  # TFRecord files
  for dataset_name in dataset_info:
    _dataset = dataset_info[dataset_name]['dataset']
    dataset_info[dataset_name]['train_path'] = _dataset.train_path
    dataset_info[dataset_name]['valid_path'] = _dataset.valid_path
    dataset_info[dataset_name]['test_path'] = _dataset.test_path

  for dataset_name in dataset_info:
    _train_path = dataset_info[dataset_name]['train_path']
    dataset_info[dataset_name]['train_dataset'] = InputDataset(_train_path, FEATURES, args.batch_size)
    _valid_path = dataset_info[dataset_name]['valid_path']
    dataset_info[dataset_name]['valid_dataset'] = InputDataset(_valid_path, FEATURES, args.batch_size)
    _test_path = dataset_info[dataset_name]['test_path']
    dataset_info[dataset_name]['test_dataset'] = InputDataset(_test_path, FEATURES, args.batch_size)

  FEATURES = {
    'targets': tf.VarLenFeature(dtype=tf.int64),
    'length': tf.FixedLenFeature([], dtype=tf.int64),
    'label': tf.FixedLenFeature([], dtype=tf.int64)
  }


  training_files = [dataset_info[dataset_name]['train_path'] for dataset_name in dataset_info]
  max_N_train = max([get_num_records(tf_rec_file) for tf_rec_file in training_files])

  # Embed words
  # TODO
  # x = tf.contrib.layers.embed_sequence(inputs,
  #                                      vocab_size=input_size,
  #                                      embed_dim=embed_dim)



  logging.info("Creating computation graph...")
  with tf.Graph().as_default():
    # Seed TensorFlow RNG
    tf.set_random_seed(args.seed)

    if args.check_numerics:
      logging.info("Checking numerics.")
      tf.add_check_numerics_ops()

    if args.print_trainable_variables:
      for tvar in tf.trainable_variables():
        print(tvar)

    # Steps per epoch
    steps_per_epoch = int(max_N_train / args.batch_size)  # cycle through datasets until all datasets have been exhaustively seen



    # Initialize model
    logging.info("Creating computation graph.")

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

    # Do training
    train_model(m, dataset_info,
                steps_per_epoch, args)


def get_train_iter(batch_size, data, args,
                   name='train_data',
                   num_epochs=None,  # None := repeat indefinitely
                   shuffle_buffer_size=10000,
                   one_shot=True,
                   prefetch_buffer_size=5000):
  # data: all training data
  with tf.name_scope(name):
    # TODO
    targets = data['text']
    labels = data['label']
    dataset = tf.data.Dataset.from_tensor_slices((targets, labels))    
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)
    if one_shot:
      iterator = dataset.make_one_shot_iterator()
    else:
      iterator = dataset.make_initializable_iterator()
  return iterator


def get_test_iter(batch_size, data, args,
                  prefetch_buffer_size=1):
  # data: all validation or test data
  with tf.name_scope('test_data'):
    targets = data['text']
    labels = data['label']
    logging.info("Creating evaluation dataset: %d images %d labels",
                 targets.shape[0], labels.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)
    iterator = dataset.make_initializable_iterator()
  return iterator


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
  # Storage for pointers to dataset-specific Tensors
  model_info = dict()
  for dataset_name in dataset_info:
    model_info[dataset_name] = dict()
  
  # Data iterators
  for dataset_name in dataset_info:
    _train_dataset = dataset_info[dataset_name]['train_dataset']
    dataset_info[dataset_name]['train_iter'] = get_train_iter(args.batch_size, _train_dataset, args, name='{}_train_dataset'.format(dataset_name))

    # Held-out test data
    if args.test:
      logging.info("Using test data for evaluation.")
      _test_dataset = dataset_info[dataset_name]['test_dataset']
      dataset_info[dataset_name]['test_iter'] = get_test_iter(args.eval_batch_size, _test_dataset, args)
    else:
      logging.info("Using validation data for evaluation.")
      _valid_dataset = dataset_info[dataset_name]['valid_dataset']
      dataset_info[dataset_name]['valid_iter'] = get_test_iter(args.eval_batch_size, _valid_dataset, args)

    # Get targets and labels
  for dataset_name in model_info:
    _train_iter = dataset_info[dataset_name]['train_iter']
    model_info[dataset_name]['targets'], model_info[dataset_name]['labels'] = _train_iter.get_next()

  # Create feature_dicts for each dataset
  for dataset_name_1 in model_info:
    _feature_dict = dict()
    for dataset_name_2 in dataset_info:
      if dataset_name_1 == dataset_name_2:
        _feature_dict[dataset_name_1] = model_info[dataset_name_1]['labels']
      else:
        _feature_dict[dataset_name_1] = None
    model_info[dataset_name]['feature_dict'] = _feature_dict

  # TODO(noa): maybe remove this 
  for dataset_name in model_info:
    _targets = model_info[dataset_name]['targets']
    _feature_dict = model_info[dataset_name]['feature_dict']
    loss = model.get_loss(_targets, _feature_dict, loss_type=???)

  # Predictions
  for dataset_name in model_info:
    _valid_iter = dataset_info[dataset_name]['valid_iter']
    model_info[dataset_name]['valid_targets'], model_info[dataset_name]['valid_labels'] = _valid_iter.get_next()
    _test_iter = dataset_info[dataset_name]['test_iter']
    model_info[dataset_name]['test_targets'], model_info[dataset_name]['test_labels'] = _test_iter.get_next()
  
    _valid_pred_op = model.get_predictions(model_info[dataset_name]['valid_targets'], dataset_info[dataset_name]['feature_name'])
    model_info[dataset_name]['valid_pred_op'] = _valid_pred_op
    _test_pred_op = model.get_predictions(model_info[dataset_name]['test_targets'], dataset_info[dataset_name]['feature_name'])
    model_info[dataset_name]['test_pred_op'] = _test_pred_op
    
  # Return dataset_info dict
  return dataset_info, model_info

def get_var_grads(loss):
  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  return (tvars, grads)


if __name__ == "__main__":
  main()
