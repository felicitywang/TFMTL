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
"""Naive Transfer Learning for document classification
1. Given tasks A and B, train two independent MLP models model A and model B with text features;
2. Predict task B on model A and get the probabilities of each label of task A;
3. Train a new model B' with text features and the results from 2 as a new feature;
4. Predict task B with model B and B'.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import shutil
import time

import tensorflow as tf
from data_prep import *
from scipy.io import mmread

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('model_dir', "./graphs/SSTb/",
                    "Directory to save tensorflow model files.")
# flags.DEFINE_string('data_dir', "../datasets/sentiment/SSTb/", "Directory of "
#                                                                "input data. ")
flags.DEFINE_string('data_dir', "../datasets/sentiment/SSTb_IMDB/SSTb/",
                    "Directory of "
                    "input data. ")
flags.DEFINE_string('model_path',
                    "../saved_models/best_IMDB",
                    "Path of the of the saved model A.")
flags.DEFINE_string('index_data_name', "data.json", "original json file with "
                                                    "index.")
flags.DEFINE_string('mtx_data_name', "data.mtx_1_3", "converted data file in "
                                                     "scipy matrix")
flags.DEFINE_string('index_name', "index.json", "List of indices of "
                                                "train/dev/test data.")
flags.DEFINE_string('report_dir', "../report/", "Directory to put the file "
                                                "report. ")
flags.DEFINE_string('report_name', "report.all", "File name of final report.")

flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_epochs', 8, 'Number of training epochs.')
flags.DEFINE_integer('hidden_layer_size', 100, 'Size of hidden layers.')
flags.DEFINE_integer('output_size', 5, 'Size of output classes.')

# flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
# flags.DEFINE_integer('total_rounds', 100, 'Max training rounds.')
# flags.DEFINE_integer('ngram', 3, 'N\'s of the n-gram model.')
# flags.DEFINE_integer('min_ngram', 1, 'Lower boundary of the range of n-values '
#                                      'for different n-grams to be extracted')
# flags.DEFINE_integer('max_ngram', 3, 'Upper boundary of the range of n-values '
#                                      'for different n-grams to be extracted')
# flags.DEFINE_integer('min_df', 50, 'Cut-off when building vocabulary.')


flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate.')
flags.DEFINE_float('lambda_', 0.0, 'L2 regularization penalty rate.')

flags.DEFINE_boolean('random_split', False, 'Whether to split data randomly '
                                            'instead of taking given indices.')

FLAGS = flags.FLAGS

# global variables
SHAPE = None


def model_fn(features, labels, mode, params):
    """MLP model using tf.estimator with 2 hidden layers and 1 dropout layer

    """

    # MLP: input -> hidden1 -> hidden2 -> dropout -> output
    input_layer = features['x']

    hidden_layer_1 = tf.layers.dense(inputs=input_layer,
                                     units=params['hidden_layer_size'],
                                     activation=tf.nn.relu)
    hidden_layer_2 = tf.layers.dense(inputs=hidden_layer_1,
                                     units=params['hidden_layer_size'],
                                     activation=tf.nn.relu)

    dropout_layer = tf.layers.dropout(
        inputs=hidden_layer_2,
        rate=params['dropout_rate'],
        # only used in TRAIN mode
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout_layer, units=params[
        'output_layer_size'],
                             activation=tf.nn.relu)

    # predictions
    pred_classes = tf.argmax(input=logits, axis=1)
    predictions = {
        'classes': pred_classes,
        'probs': tf.nn.softmax(logits, name='softmax_tensor')
    }
    # used for export_savedmodel
    export_outputs = {
        tf.saved_model.signature_constants.PREDICT_METHOD_NAME: tf.estimator.export.PredictOutput(
            {
                'classes': predictions['classes'],
                'probs': predictions['probs']
            }),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

    # l2 regularization
    variables = tf.trainable_variables()
    weight_penalty = tf.add_n([tf.nn.l2_loss(v) for v in variables
                               if 'bias' not in v.name]) * params[
                         'lambda_']
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

    # for dev mode
    # evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=pred_classes),
        "precision": tf.metrics.precision(labels=labels,
                                          predictions=pred_classes),
        "recall": tf.metrics.recall(labels=labels,
                                    predictions=pred_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn():
    """Used for tf.estimator.Estimator.export_savedmodel

    """
    feature_spec = {'x': tf.placeholder(dtype=tf.float64, shape=[
        FLAGS.batch_size, SHAPE])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
        features=feature_spec)()


# TODO precision,recall,f1
def compute_accuracy(pred, gold):
    assert len(pred) == len(gold)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gold[i]:
            correct += 1
    return 1.0 * correct / len(pred)


def load_data():
    data_dir = FLAGS.data_dir

    x_train = mmread(data_dir + "x_train.mtx")
    x_dev = mmread(data_dir + "x_dev.mtx")
    x_test = mmread(data_dir + "x_test.mtx")

    x_train = x_train.tocsr()
    x_dev = x_dev.tocsr()
    x_test = x_test.tocsr()

    y = np.load(data_dir + "y.npz")
    y_train = y['arr_0']
    y_dev = y['arr_1']
    y_test = y['arr_2']

    return x_train, y_train, x_dev, y_dev, x_test, y_test

    # transform if file doesn't yet exist
    # mtx_path = pathlib.Path(FLAGS.data_dir + FLAGS.mtx_data_name)
    # if not mtx_path.is_file():
    #     transform_data(FLAGS.data_dir, FLAGS.index_data_name,
    #                    FLAGS.mtx_data_name, FLAGS.min_ngram,
    #                    FLAGS.max_ngram,
    #                    FLAGS.min_df)
    # if FLAGS.random_split:
    #     return random_split_data(
    #         FLAGS.data_dir, FLAGS.index_data_name, FLAGS.mtx_data_name)
    # else:
    #     index_dict = json.loads(
    #         open(FLAGS.data_dir + FLAGS.index_name).read())
    #     return split_data(FLAGS.data_dir, FLAGS.index_data_name,
    #                       FLAGS.mtx_data_name, index_dict)


def test_model(model_path, x_test, y_test, pred_test):
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   model_path)
        predictor = tf.contrib.predictor.from_saved_model(
            model_path)
        output_dict = predictor(
            {'x': np.hstack((x_test.toarray(), pred_test))})
        pred_classes = output_dict['classes']

        # compute accuracy
        return compute_accuracy(pred_classes, y_test)


def predict(model_path, x):
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   model_path)
        predictor = tf.contrib.predictor.from_saved_model(
            model_path)
        output_dict = predictor({'x': x.toarray()})
        return output_dict['probs']


class MLP(object):
    """
    MLP class with model, tran/test and some helper methods
    """

    def __init__(self, params, report_name):
        self.params = params
        self.report_name = report_name

        # MLP model using tf.estimator
        self.model = tf.estimator.Estimator(model_fn=model_fn,
                                            model_dir=FLAGS.model_dir,
                                            params=self.params)

        self.best_model_path = None

    def train(self, x_train, y_train, x_dev, y_dev, pred_train, pred_dev):
        """Train

        1. train several epochs on the training set
        2. evaluate on dev set
        3. return if loss on dev set stops decreasing; repeat 1,2 otherwise

        Always save last checkpoint path so as to restore last(best)
        hyperparameters after model stops improving on dev set
        """

        # TRAIN mode input: train on training set for certain epochs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.hstack((x_train.toarray(), pred_train))},
            y=y_train,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_epochs=2
        )

        # dev mode input: evaluate on dev set, run once
        dev_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.hstack((x_dev.toarray(), pred_dev))},
            y=y_dev,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_epochs=1
        )

        print("Start training...")

        # evaluation metrics: accuracy / loss on dev set
        # stop training when loss on dev set stops decreasing

        max_accy = 0
        best_epoch = 0

        # cross validation
        # for train_index, dev_index in kfold_dev.split(X, y):
        # print("//////////fold %d//////////" % i)
        # i += 1

        # instead of early stopping, save results on dev set for each epoch and
        # corresponding checkpoint, use the parameters for the best result on
        # dev set

        for epoch in range(self.params['num_epochs']):
            self.model.train(train_input_fn)
            dev = self.model.evaluate(dev_input_fn)
            dev_accy = dev['accuracy']
            dev_loss = dev['loss']

            print("########## epoch ", epoch)
            print("########## accy = %.6f " % dev_accy)
            print("########## loss = %.6f " % dev_loss)

            if dev_accy > max_accy:
                best_epoch = epoch
                max_accy = dev_accy

                if self.best_model_path is not None and os.stat(
                        self.best_model_path):
                    shutil.rmtree(self.best_model_path)

                self.best_model_path = self.model.export_savedmodel(
                    export_dir_base=FLAGS.model_dir + "accy/",
                    serving_input_receiver_fn=serving_input_receiver_fn,
                    checkpoint_path=tf.train.latest_checkpoint(
                        checkpoint_dir=FLAGS.model_dir))

        print("*********** best epoch(accy) = %d / %d" % (best_epoch + 1,
                                                          FLAGS.num_epochs))

    def test(self, x_test, y_test, pred_test):
        """evaluate on test set using the saved model

        """

        test_accy = test_model(self.best_model_path, x_test,
                               y_test, pred_test)

        print("test accy = %f%%" % (100. * test_accy))

        # write all the results of all the hyperparameters/data into one file
        with open(self.report_name, "a") as file:
            file.write("|" + str(FLAGS.learning_rate))
            file.write("|" + str(FLAGS.hidden_layer_size))
            file.write("|" + str(FLAGS.batch_size))
            file.write("|" + str(FLAGS.dropout_rate))
            file.write("|" + str(FLAGS.lambda_))
            file.write("|" + str(test_accy))
            file.write("|" + "?")
            file.write("|" + "?")
            file.write("|" + "?")

            file.write("\n")

            file.write(
                "best model path: " + self.best_model_path.decode(
                    'utf-8'))
            file.write("\n")

            file.close()


def main(argv):
    # report name
    report_name = FLAGS.report_dir + FLAGS.report_name
    report_path = pathlib.Path(report_name)
    if not report_path.is_file():
        with open(report_name, "a") as file:
            file.write("learning rate|hidden layer size|batch size"
                       "|dropout rate|l2 "
                       "lambda|loss|accy(accy)|precision"
                       "|recall|f1\n")
            file.write("---|---|---|---|---|---|---|---|---|---\n")

    # load data
    ticks = time.clock()
    # x_train, y_train, x_dev, y_dev, x_test, y_test, OUTPUT_SIZE \
    #     = load_data()
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data()
    print("Data loaded and split in %.6f s" % (time.clock() - ticks))

    # predict on another model and add the new features
    pred_train = predict(FLAGS.model_path, x=x_train)
    pred_dev = predict(FLAGS.model_path, x=x_dev)
    pred_test = predict(FLAGS.model_path, x=x_test)

    # shape used for serving_input_receiver_fn
    global SHAPE
    SHAPE = x_train.shape[1] + len(pred_train[0])

    # hyperparameters for MLP model
    params = {'lambda_': FLAGS.lambda_,
              'learning_rate': FLAGS.learning_rate,
              'dropout_rate': FLAGS.dropout_rate,
              'hidden_layer_size': FLAGS.hidden_layer_size,
              'output_layer_size': FLAGS.output_size,
              'num_epochs': FLAGS.num_epochs
              }

    mlp = MLP(params=params, report_name=report_name)

    ticks = time.clock()
    mlp.train(x_train, y_train, x_dev, y_dev, pred_train, pred_dev)
    print("Data trained in %.6f s" % (time.clock() - ticks))

    ticks = time.clock()
    mlp.test(x_test, y_test, pred_test)
    print("Data predicted in %.6f s" % (time.clock() - ticks))


if __name__ == "__main__":
    tf.app.run()
