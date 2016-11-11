# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A library to evaluate nn on mnist validation data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time

import sys
import numpy as np
import tensorflow as tf

import mnist

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def do_eval(saver,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))
    else:
      print('No checkpoint file found')
      sys.stdout.flush()
      return

    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, global_step))
    sys.stdout.flush()

    # Let batch size be equal to the whole evaluation data
    num_examples = data_set.num_examples
    feed_dict = mnist.fill_feed_dict(data_set,
                                     images_placeholder,
                                     labels_placeholder)
    true_count = sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    sys.stdout.flush()

def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    batch_size = dataset.num_examples()
    images_placeholder, labels_placeholder = mnist.placeholder_inputs(batch_size)
    logits, reg = mnist.inference(images_placeholder, train=False)
    eval_correct = mnist.evaluation(logits, labels_placeholder)
    sess = tf.Session()
    saver = tf.train.Saver()
    while True:
      do_eval(saver, eval_correct, images_placeholder, labels_placeholder, dataset)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
