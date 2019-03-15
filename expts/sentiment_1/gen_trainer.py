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
from collections import Counter
from collections import defaultdict
from itertools import product
import numpy as np
import tensorflow as tf

from decoder_factory import build_decoders

from mtl.tf_io import get_num_records
from mtl.tf_io import get_empirical_label_prior
from mtl.hparams import update_hparams_from_args
from mtl.util.pipeline import Pipeline

from mtl.embedders.embed_sequence import embed_sequence
from mtl.extractors.cnn import conv_and_pool

from mtl.mlvae.simple_mlvae_model import default_hparams as simple_mlvae_hparams
from mtl.mlvae.simple_mlvae_model import SimpleMultiLabelVAE

from mtl.mlvae.mlvae_model import default_hparams as normal_mlvae_hparams
from mtl.mlvae.mlvae_model import MultiLabelVAE

from mtl.mlvae.joint_mlvae_model import default_hparams as joint_mlvae_hparams
from mtl.mlvae.joint_mlvae_model import JointMultiLabelVAE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging = tf.logging

FEATURES = {
    'label': tf.FixedLenFeature([], dtype=tf.int64),
    'types': tf.VarLenFeature(dtype=tf.int64),
    'type_counts': tf.VarLenFeature(dtype=tf.int64),
    'types_length': tf.FixedLenFeature([], dtype=tf.int64),
    'tokens': tf.VarLenFeature(dtype=tf.int64),
    'tokens_length': tf.FixedLenFeature([], dtype=tf.int64),
}

IMDB_NUM_LABEL = 2
SSTb_NUM_LABEL = 5


def parse_args():
    p = ap.ArgumentParser()
    p.add_argument('--model', type=str, default='normal',
                   choices=['simple', 'normal', 'joint'], help='Model to use.')
    p.add_argument('--test', action='store_true', default=False,
                   help='Use held-out test data. WARNING: DO NOT TUNE ON TEST')
    p.add_argument('--batch-size', default=128, type=int,
                   help='Size of batch.')
    p.add_argument('--unlabeled', action='store_true', dest='unlabeled',
                   help="Use unlabeled data")
    p.add_argument('--no-unlabeled', action='store_false', dest='unlabeled',
                   help="No unlabeled data")
    p.set_defaults(unlabeled=False)
    p.add_argument('--alpha', default=0.50, type=float,
                   help='Weight placed on the discriminative terms [0, 1.0]')
    p.add_argument('--encoder-arch', default="conv_max_tied",
                   choices=["conv_max_tied", "conv_max_untied"],
                   help="Type of encoder to use for each task.")
    p.add_argument('--decoder-arch', default="bow_tied",
                   choices=["bow_untied", "bow_tied", "cnn_unigram",
                            "shallow_cnn_unigram"],
                   help="Type of decoder to use for each task.")
    p.add_argument('--y-prediction', default="deterministic",
                   choices=['deterministic', 'sampled'],
                   help="Method for test-time prediction")
    p.add_argument('--y-inference', default="exact",
                   choices=['exact', 'sample'],
                   help="How to infer latent labels")
    p.add_argument('--label-prior-type', default="uniform",
                   choices=["uniform", "learned", "fixed"],
                   help="What type of label prior to use")
    p.add_argument('--eval-batch-size', default=256, type=int,
                   help='Size of evaluation batch.')
    p.add_argument('--word-embed-dim', default=256, type=int,
                   help='Word embedding size')
    p.add_argument('--latent-dim', default=128, type=int,
                   help='Latent embedding dimensionality')
    p.add_argument('--mlp-hidden-dim', default=512, type=int,
                   help='Size of the MLP hidden layers.')
    p.add_argument('--mlp-num-layers', default=2, type=int,
                   help='Number of MLP layers')
    p.add_argument('--share-embed', action='store_true', dest='share-embed',
                   help='Share word embeddings between tasks')
    p.add_argument('--no-share-embed', action='store_false', dest='share-embed',
                   help='Do not share word embeddings between tasks')
    p.set_defaults(share_embed=True)
    p.add_argument('--share-decoders', action='store_true', dest='share-decoder',
                   help='Share decoders between tasks')
    p.add_argument('--no-share-decoders', action='store_false',
                   dest='share-decoder',
                   help='Share decoders between tasks')
    p.set_defaults(share_decoder=False)
    p.add_argument('--layer-norm', action='store_true', dest='layer-norm',
                   help='Use layer normalization')
    p.add_argument('--no-layer-norm', action='store_false', dest='layer-norm',
                   help='No layer normalization')
    p.set_defaults(layer_norm=True)
    p.add_argument('--qy-mlp-hidden-dim', default=512, type=int,
                   help='[qy] MLP hidden dim')
    p.add_argument('--qy-mlp-num-layer', default=2, type=int,
                   help='[qy] MLP number of hidden layers')
    p.add_argument('--qz-mlp-hidden-dim', default=512, type=int,
                   help='[qz] MLP hidden dim')
    p.add_argument('--qz-mlp-num-layer', default=2, type=int,
                   help='[qz] MLP number of hidden layers')
    p.add_argument('--pz-mlp-hidden-dim', default=512, type=int,
                   help='[pz] MLP hidden dim')
    p.add_argument('--pz-mlp-num-layer', default=2, type=int,
                   help='[pz] MLP number of hidden layers')
    p.add_argument('--lr0', default=0.0001, type=float,
                   help='Initial learning rate')
    p.add_argument('--max-grad-norm', default=5.0, type=float,
                   help='Clip gradients to max_grad_norm during training.')
    p.add_argument('--num-train-epochs', default=100, type=int,
                   help='Number of training epochs.')
    p.add_argument('--print-trainable-variables', action='store_true',
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


def get_joint_log_prior(LMRD_path, SSTb_path):
    LMRD_labels = [0, 1]
    SSTb_labels = [0, 1, 2, 3, 4]
    joint_dist = np.zeros([len(LMRD_labels) * len(SSTb_labels)])
    index = 0
    for e in product(LMRD_labels, SSTb_labels):
        LMRD_label = e[0]
        SSTb_label = e[1]
        if SSTb_label > 2:
            mapped_SSTb_label = 1
        elif SSTb_label < 2:
            mapped_SSTb_label = 0
        else:
            mapped_SSTb_label = 2

        if mapped_SSTb_label == LMRD_label:
            joint_dist[index] = 2.0
        elif mapped_SSTb_label == 2:
            joint_dist[index] = 1.0
        else:
            joint_dist[index] = 0.1

        index += 1

    joint_dist = joint_dist / np.sum(joint_dist)
    return np.log(joint_dist)


def encoder_graph(batch, embed_fn, tokens_key='tokens', max_filter_width=7):
    tokens = batch[tokens_key]
    embed = embed_fn(tokens)
    return conv_and_pool(embed, num_filter=256, max_width=max_filter_width)


def build_encoders(arch, vocab_size, args, embedder):
    encoders = dict()

    if arch == "conv_max_tied":
        feature_extractor = tf.make_template('encoder', encoder_graph,
                                             embed_fn=embedder)
        for ds in args.datasets:
            encoders[ds] = feature_extractor
    elif arch == "conv_max_untied":
        for ds in args.datasets:
            encoders[ds] = tf.make_template('encoder_%s' % ds, encoder_graph,
                                            embed_fn=embedder)
    else:
        raise ValueError("unrecognized encoder architecture")

    return encoders


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
    loss = model.get_multi_task_loss(train_batches)
    SSTb_g_loss = model.get_generative_loss('SSTb')
    IMDB_g_loss = model.get_generative_loss('IMDB')
    SSTb_d_loss = model.get_discriminative_loss('SSTb')
    IMDB_d_loss = model.get_discriminative_loss('IMDB')
    SSTb_loss = model.get_task_loss('SSTb')
    IMDB_loss = model.get_task_loss('IMDB')

    # Done building compute graph; set up training ops.

    # Training ops
    global_step_tensor = tf.train.get_or_create_global_step()
    lr = get_learning_rate(args.lr0)
    tvars, grads = get_var_grads(loss, args)
    train_op = get_train_op(tvars, grads, lr, args.max_grad_norm,
                            global_step_tensor, name='train_op')
    init_ops = [tf.global_variables_initializer(),
                tf.local_variables_initializer()]
    config = get_proto_config(args)
    with tf.train.SingularMonitoredSession(config=config) as sess:
        tf.set_random_seed(args.seed)

        # Initialize model parameters and optimizer operations
        sess.run(init_ops)

        # Do training
        for epoch in xrange(args.num_train_epochs):
            start_time = time()

            events = {}
            events['SSTb'] = Counter()
            events['IMDB'] = Counter()

            # Take steps_per_epoch gradient steps
            total_loss = 0.0
            total_g_loss = defaultdict(float)
            total_d_loss = defaultdict(float)
            total_SSTb_loss = 0.0
            total_IMDB_loss = 0.0
            num_iter = 0
            for i in xrange(steps_per_epoch):
                vals = sess.run([
                    global_step_tensor,
                    loss,
                    SSTb_g_loss,
                    IMDB_g_loss,
                    SSTb_d_loss,
                    IMDB_d_loss,
                    SSTb_loss,
                    IMDB_loss,
                    model.obs_label('SSTb'),
                    model.obs_label('IMDB'),
                    model.latent_preds('SSTb', 'IMDB'),
                    model.latent_preds('IMDB', 'SSTb'),
                    train_op])

                num_iter += 1
                step = vals[0]
                total_loss += vals[1]
                total_g_loss['SSTb'] += vals[2]
                total_g_loss['IMDB'] += vals[3]
                total_d_loss['SSTb'] += vals[4]
                total_d_loss['IMDB'] += vals[5]
                total_SSTb_loss += vals[6]
                total_IMDB_loss += vals[7]

                # Update joint dists
                SSTb_obs = ['obs_%d' % x for x in vals[8].tolist()]
                SSTb_lat = ['lat_%d' % x for x in vals[10].tolist()]
                events['SSTb'] += Counter(zip(SSTb_obs, SSTb_lat))

                IMDB_obs = ['obs_%d' % x for x in vals[9].tolist()]
                IMDB_lat = ['lat_%d' % x for x in vals[11].tolist()]
                events['IMDB'] += Counter(zip(IMDB_obs, IMDB_lat))

            assert num_iter > 0

            # average loss per batch (which is in turn averaged across examples)
            train_loss = total_loss / float(num_iter)
            train_SSTb_loss = total_SSTb_loss / float(num_iter)
            train_IMDB_loss = total_IMDB_loss / float(num_iter)

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
                str_ = '[epoch=%d/%d step=%d (%d s)] loss=%.2f' % (
                    epoch + 1, args.num_train_epochs, np.asscalar(step), elapsed,
                    train_loss)
                str_ += ' SSTb_loss=%.2f IMDB_loss=%.2f' % (train_SSTb_loss,
                                                            train_IMDB_loss)
                for dataset_name in model_info:
                    g_loss = total_g_loss[dataset_name] / float(num_iter)
                    d_loss = total_d_loss[dataset_name] / float(num_iter)
                    str_ += ' %s_g_loss=%.2f %s_d_loss=%.4f' % (dataset_name,
                                                                g_loss,
                                                                dataset_name,
                                                                d_loss)

                tot_eval_acc = 0.0
                for dataset_name in model_info:
                    num_eval_total = model_info[dataset_name]['test_metrics']['ntotal']
                    eval_acc = model_info[dataset_name]['test_metrics']['accuracy']
                    tot_eval_acc += eval_acc
                    str_ += '\n(%s) num_eval_total=%d eval_acc=%f' % (
                        dataset_name, num_eval_total, eval_acc)

                    str_ += '\nMost frequent events:'
                    for event, count in events[dataset_name].most_common(5):
                        str_ += '\n%s: %d' % (event, count)

                str_ += '\nmean_eval_acc=%f' % (tot_eval_acc / float(len(model_info)))
                logging.info(str_)
            else:
                raise "final evaluation mode not implemented"


def compute_held_out_performance(session, pred_op, eval_label,
                                 eval_iterator, args):
    # Initialize the iterator
    session.run(eval_iterator.initializer)

    # Accumulate predictions
    gold_labels = []
    pred_labels = []
    while True:
        try:
            y, y_hat = session.run([eval_label, pred_op])
            assert y.shape == y_hat.shape
            gold_labels += y.tolist()
            pred_labels += y_hat.tolist()
        except tf.errors.OutOfRangeError:
            break

    assert len(gold_labels) == len(pred_labels)

    ntotal = len(gold_labels)
    ncorrect = 0
    for i in xrange(len(gold_labels)):
        if gold_labels[i] == pred_labels[i]:
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
        embedder = tf.make_template('embedding', embed_sequence,
                                    vocab_size=vocab_size,
                                    embed_dim=args.word_embed_dim)
        encoders = build_encoders(args.encoder_arch, vocab_size, args,
                                  embedder=embedder)

        # Regularization
        decoders = build_decoders(args.decoder_arch, vocab_size, args,
                                  embedder=embedder)

        # Maybe check for NaNs
        if args.check_numerics:
            logging.info("Checking numerics.")
            tf.add_check_numerics_ops()

        # Maybe print out trainable variables of the model
        if args.print_trainable_variables:
            for tvar in tf.trainable_variables():
                # TODO: also print and sum up all their sizes
                print(tvar)

        if args.model == 'simple':
            hp = simple_mlvae_hparams()
            update_hparams_from_args(hp, args)
            m = SimpleMultiLabelVAE(class_sizes=class_sizes, encoders=encoders,
                                    decoders=decoders, hp=hp)
        elif args.model == 'normal':
            hp = normal_mlvae_hparams()
            update_hparams_from_args(hp, args)
            m = MultiLabelVAE(class_sizes=class_sizes, encoders=encoders,
                              decoders=decoders, hp=hp)
        elif args.model == 'joint':
            hp = joint_mlvae_hparams()
            update_hparams_from_args(hp, args)

            if args.label_prior_type == 'fixed':
                prior = get_joint_log_prior(dataset_info['IMDB']['train_path'],
                                            dataset_info['SSTb']['train_path'])
            else:
                prior = None

            m = JointMultiLabelVAE(class_sizes=class_sizes, encoders=encoders,
                                   decoders=decoders, prior=prior, hp=hp)
        else:
            raise ValueError("unrecognized model: %s" % (args.model))

        # Do training
        tf.logging.info("Batch size: %d", args.batch_size)
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
        _train_batch = _train_dataset.batch
        model_info[dataset_name]['train_iter'] = _train_iter
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
            logging.info("[%s] WARNING: Using test data for evaluation.",
                         dataset_name)
        else:
            logging.info("[%s] Using validation data for evaluation.", dataset_name)

    # Return dataset_info dict and model_info dict
    return dataset_info, model_info


def build_input_dataset(tfrecord_path, batch_features, batch_size,
                        is_training=True):
    if is_training:
        ds = Pipeline(tfrecord_path, batch_features, batch_size,
                      one_shot=True, num_epochs=None)
    else:
        ds = Pipeline(tfrecord_path, batch_features, batch_size,
                      num_epochs=1, one_shot=False)
    return ds


def get_var_grads(loss, args):
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
    return (tvars, grads)


if __name__ == "__main__":
    main()
