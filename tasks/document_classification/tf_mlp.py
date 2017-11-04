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
"""A 2-hidden-layer MLP model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from sklearn_prepare import *

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('model_dir', "./mlp/", 'Model directory.')
flags.DEFINE_string('mtx_data_name', "X_data_50_parent_text.mtx",
                    "File name of preprocessed data in scipy mtx format.")
flags.DEFINE_string('index_data_name', "post_df_parent_text.json",
                    "File name of X, y with indices for GroupKFold")
flags.DEFINE_string('base_dir', "../", "Directory of data/ and "
                                       "models/ directories.")

flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs.')
# flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('total_rounds', 200, 'Max training rounds.')
flags.DEFINE_integer('hidden_layer_size', 200, 'Size of hidden layers.')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_float('lambda_', 0.0, 'L2 regularization penalty rate.')

FLAGS = flags.FLAGS

# model parameters
OUTPUT_SIZE = 9

# global variables
model = None
last_checkpoint_path = None
X_train, y_train = None, None
X_dev, y_dev = None, None
X_test, y_test = None, None


def model_fn(features, labels, mode, params):
    """MLP model using tf.estimator with 2 hidden layers and 1 dropout layer

    """

    # MLP: input -> hidden1 -> hidden2 -> dropout -> output
    input_layer = features['all']

    hidden_layer_1 = tf.layers.dense(inputs=input_layer,
                                     units=params['hidden_layer_size'],
                                     activation=tf.nn.relu)
    hidden_layer_2 = tf.layers.dense(inputs=hidden_layer_1,
                                     units=params['hidden_layer_size'],
                                     activation=tf.nn.relu)

    dropout_layer = tf.layers.dropout(
        inputs=hidden_layer_2,
        rate=params['dropout_rate'],
        training=mode == tf.estimator.ModeKeys.TRAIN)  # only used in TRAIN mode

    logits = tf.layers.dense(inputs=dropout_layer, units=OUTPUT_SIZE,
                             activation=tf.nn.relu)

    # predictions
    predictions = tf.argmax(logits, axis=1)

    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

    # l2 regularization
    variables = tf.trainable_variables()
    weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in variables
                               if 'bias' not in v.name]) * params['lambda_']
    loss_op = loss + weight_penalty

    # for TRAIN mode
    # accuracy
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss_op,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op,
                                          train_op=train_op)

    # for PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # for EVAL mode
    # evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions),
        "precision": tf.metrics.precision(labels=labels,
                                          predictions=predictions),
        "recall": tf.metrics.recall(labels=labels,
                                    predictions=predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


"""Split data into train, dev, test sets

currently train:dev:test = 64 : 16 : 20
"""


def train():
    """Train

    1. train several epochs on the training set
    2. evaluate on dev set
    3. return if loss on dev set stops decreasing; repeat 1,2 otherwise

    Always save last checkpoint path so as to restore last(best)
    hyperparameters after model stops improving on dev set
    """
    global model
    global last_checkpoint_path
    global X_train, y_train, X_dev, y_dev

    # TRAIN mode input: train on training set for certain epochs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'all': X_train.toarray()}, y=y_train,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_epochs=FLAGS.num_epochs
    )

    # DEV mode input: evaluate on dev set, run once
    dev_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'all': X_dev.toarray()}, y=y_dev,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_epochs=1
    )

    print("Start training...")

    # evaluation metrics: accuracy / loss on dev set
    # stop training when loss on dev set stops decreasing
    dev_accuracy = 0
    prev_dev_accuracy = 0

    dev_loss = 0
    prev_dev_loss = sys.float_info.max

    # cross validation
    # for train_index, dev_index in kfold_dev.split(X, y):
    # print("//////////fold %d//////////" % i)
    # i += 1

    for round_ in range(FLAGS.total_rounds):
        print("##########round", round_)
        # model.train(train_input_fn, steps=FLAGS.num_steps)
        model.train(train_input_fn)
        # dev = model.evaluate(dev_input_fn, steps=200)
        dev = model.evaluate(dev_input_fn)
        dev_accuracy = dev['accuracy']
        dev_loss = dev['loss']
        tol = 0.00001

        # if accuracy < prev_accuracy:
        # if (accuracy - prev_accuracy) <= tol:
        #     break
        if dev_loss - prev_dev_loss >= tol:
            break
        print(
            "##########accuracy=%.6f(prev=%.6f), "
            "##########loss=%.6f(prev=%.6f), "
            "continue training... "
            % (dev_accuracy, prev_dev_accuracy, dev_loss, prev_dev_loss))
        prev_dev_accuracy = dev_accuracy
        prev_dev_loss = dev_loss
        last_checkpoint_path = tf.train.latest_checkpoint(
            checkpoint_dir=FLAGS.model_dir)

    print(
        "**********accuracy=%.6f(prev=%.6f),accuracy "
        "**********loss=%.6f(prev=%.6f),loss "
        "stopped improving, "
        "training done"
        % (dev_accuracy, prev_dev_accuracy, dev_loss, prev_dev_loss))


def test():
    """Evaluate on test set using hyperparameters stored in last checkpoint

    """
    global model
    global last_checkpoint_path
    global X_test, y_test

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'all': X_test.toarray()}, y=y_test,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_epochs=1
    )

    evaluate = model.evaluate(test_input_fn,
                              checkpoint_path=last_checkpoint_path)
    print("**********test accuracy, precision, recall = "
          "%.2f, %2f, %.2f "
          % (evaluate['accuracy'], evaluate['precision'],
             evaluate['recall']))

    # write all results of all hyperparameters into one file
    report_name = FLAGS.base_dir + "models/report.all"

    with open(report_name, "a") as file:
        file.write("\n\nlearning rate = " + str(FLAGS.learning_rate))
        file.write("\nhidden_layer_size = " + str(FLAGS.hidden_layer_size))
        file.write("\nbatch_size = " + str(FLAGS.batch_size))
        file.write("\ndropout_rate = " + str(FLAGS.dropout_rate))
        file.write("\nlambda_ = " + str(FLAGS.lambda_))

        file.write("\n\ntest loss = " + str(evaluate['loss']))
        file.write("\ntest accuracy = " + str(evaluate['accuracy']))
        file.write("\ntest precision = " + str(evaluate['precision']))
        file.write("\ntest recall = " + str(evaluate['recall']))

        file.close()


def main(self):
    global X_train, y_train, X_dev, y_dev, X_test, y_test
    ticks = time.clock()
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_data(
        FLAGS.base_dir, FLAGS.index_data_name, FLAGS.mtx_data_name)
    print("Data loaded and split in %.2f s" % (time.clock() - ticks))

    # hyperparameters for MLP model
    params = {'lambda_': FLAGS.lambda_,
              'learning_rate': FLAGS.learning_rate,
              'dropout_rate': FLAGS.dropout_rate,
              'hidden_layer_size': FLAGS.hidden_layer_size,
              }

    # MLP model using tf.estimator
    global model
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=FLAGS.model_dir,
                                   params=params)
    ticks = time.clock()
    train()
    print("Data trained in %.2f s" % (time.clock() - ticks))

    ticks = time.clock()
    test()
    print("Data trained in %.2f s" % (time.clock() - ticks))


if __name__ == "__main__":
    tf.app.run()
