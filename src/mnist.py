# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import math

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import logging_ops

# The MNIST images are always 28x28 pixels.
NUM_LABELS = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
SEED = 66478  # Set to None for random seed.

FLAGS = tf.app.flags.FLAGS

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size,))
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def inference(images, train=True):

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=tf.float32))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1,
      seed=SEED, dtype=tf.float32))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=tf.float32))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=tf.float32))

  tf.logging.Info(conv1_weights.device)
  tf.logging.Info(conv1_biases.device)
  tf.logging.Info(conv2_weights.device)
  tf.logging.Info(conv3_biases.device)
  tf.logging.Info(fc1_weights.device)
  tf.logging.Info(fc1_biases.device)
  tf.logging.Info(fc2_weights.device)
  tf.logging.Info(fc2_biases.device)

  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  conv = tf.nn.conv2d(images,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  # Bias and rectified linear non-linearity.
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
  # Max pooling. The kernel size spec {ksize} also follows the layout of
  # the data. Here we have a pooling window of 2, and a stride of 2.
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  conv = tf.nn.conv2d(pool,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  pool_shape = pool.get_shape().as_list()
  reshape = tf.reshape(
      pool,
      [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

  reg = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
         tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

  logits = tf.matmul(hidden, fc2_weights) + fc2_biases

  return logits, 5e-4 * reg

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

def evaluation(logits, labels):
  pred = tf.nn.softmax(logits)
  correct = tf.nn.in_top_k(pred, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def predictions(logits):
    return tf.nn.softmax(logits)
