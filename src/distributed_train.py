# This code is taken and modified from the inception_distribute_train.py file of
# google's tensorflow inception model. The original source is here - https://github.com/tensorflow/models/tree/master/inception.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from threading import Timer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import input as tf_input

import cifar_input
import resnet_model

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('n_train_epochs', 1000, 'Number of epochs to train for')
tf.app.flags.DEFINE_boolean('should_summarize', False, 'Whether Chief should write summaries.')
tf.app.flags.DEFINE_boolean('timeline_logging', False, 'Whether to log timeline of events.')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for timeout communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('variable_batchsize_r', False,
                            'Use variable batchsize comptued using R.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 300,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
#tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
#                          'Initial learning rate.')
# For flowers
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

EVAL_BATCHSIZE=2000

def compute_train_error(sess, top_k_op, epoch, dq, images_pl, labels_pl, e_time):
  train_error_start_time = time.time()
  step = 0
  #batch_size = FLAGS.batch_size
  batch_size = 1024
  num_iter = int(math.ceil(60000 / batch_size))
  true_count = 0  # Counts the number of correct predictions.
  total_sample_count = num_iter * batch_size
  while step < num_iter:
    t1 = time.time()
    images_real, labels_real = sess.run(dq)
    t2 = time.time()
    feed_dict = cifar_input.fill_feed_dict(images_real, labels_real, images_pl, labels_pl)
    t3 = time.time()
    predictions = sess.run([top_k_op], feed_dict=feed_dict)
    t4 = time.time()
    tf.logging.info("Images time: %f, feed_dict time: %f, predictions time: %f" % (t2-t1, t3-t2, t4-t3))
    true_count += np.sum(predictions)
    step += 1
  train_error_end_time = time.time()
  precision = true_count / total_sample_count
  tf.logging.info('Epoch %f %f %f' % (e_time, epoch, precision))
  sys.stdout.flush()

def compute_R(sess, grads_and_vars, dq, images_pl, labels_pl):
  step = 0
  num_iter = int(math.ceil(60000 / float(1024)))
  sum_of_norms, norm_of_sums = None, None
  while step < num_iter:
    images_real, labels_real = sess.run(dq)

    feed_dict = cifar_input.fill_feed_dict(images_real, labels_real, images_pl, labels_pl)

    gradients = sess.run([x[0] for x in grads_and_vars], feed_dict=feed_dict)
    gradient = np.concatenate(np.array([x.flatten() for x in gradients]))
    gradient *= FLAGS.batch_size

    if sum_of_norms == None:
      sum_of_norms = np.linalg.norm(gradient)**2
    else:
      sum_of_norms += np.linalg.norm(gradient)**2

    if norm_of_sums == None:
      norm_of_sums = gradient
    else:
      norm_of_sums += gradient

    step += 1

  ratio = num_iter * 1024 * sum_of_norms / np.linalg.norm(norm_of_sums)**2
  tf.logging.info("batchsize ratio: %f" % ratio)
  return ratio

def train(target, cluster_spec):

  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are infered from the workers and ps
  # hosts string.
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  is_chief = (FLAGS.task_id == 0)

  # Ops are assigned to worker by default.
  with tf.device(
      tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % FLAGS.task_id,
        cluster=cluster_spec)):

    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables. The PS holds the global step.

    images, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size, "train")
    #images, labels = cifar_input.placeholder_inputs()
    variable_batchsize_inputs = cifar_input.build_input_multi_batchsize(FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size, "train")

    hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                               num_classes=10 if FLAGS.dataset=="cifar10" else 100,
                               min_lrn_rate=0.0001,
                               lrn_rate=FLAGS.initial_learning_rate,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='sgd')

    model = resnet_model.ResNet(hps, images, labels, "train")
    model.build_graph()

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate)

    # Use V2 optimizer
    opt = tf.train.SyncReplicasOptimizer(
      opt,
      num_workers,
      total_num_replicas=num_workers,
    )

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(model.cost)
    apply_gradients_op = opt.apply_gradients(grads, global_step=model.global_step)

    with tf.control_dependencies([apply_gradients_op]):
      train_op = tf.identity(model.cost, name='train_op')

    # Get chief queue_runners, init_tokens and clean_up_op, which is used to
    # synchronize replicas.
    # More details can be found in sync_replicas_optimizer.
    chief_queue_runners = [opt.get_chief_queue_runner()]
    init_tokens_op = opt.get_init_tokens_op()
    #clean_up_op = opt.get_clean_up_op()

    # Create a saver.
    saver = tf.train.Saver(max_to_keep=100)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init_op = tf.initialize_all_variables()

    test_print_op = logging_ops.Print(0, [0], message="Test print success")

    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    if is_chief:
      local_init_op = opt.chief_init_op
    else:
      local_init_op = opt.local_step_init_op

    local_init_opt = [local_init_op]
    ready_for_local_init_op = opt.ready_for_local_init_op

    sv = tf.train.Supervisor(is_chief=is_chief,
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             logdir=FLAGS.train_dir,
                             init_op=init_op,
                             summary_op=None,
                             global_step=model.global_step,
                             saver=saver,
                             save_model_secs=FLAGS.save_interval_secs,
    )

    tf.logging.info("BATCHSIZE: %d" % FLAGS.batch_size);

    tf.logging.info('%s Supervisor' % datetime.now())

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)

    # Get a session.
    sess = sv.prepare_or_wait_for_session(target, config=sess_config)

    # Start the queue runners.
    queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    sv.start_queue_runners(sess, queue_runners)
    tf.logging.info('Started %d queues for processing input data.',
                    len(queue_runners))

    if is_chief:
      sv.start_queue_runners(sess, chief_queue_runners)
      sess.run(init_tokens_op)

    # Train, checking for Nans. Concurrently run the summary operation at a
    # specified interval. Note that the summary_op and train_op never run
    # simultaneously in order to prevent running out of GPU memory.
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    begin_time = time.time()

    # Keep track of own iteration
    cur_iteration = -1
    iterations_finished = set()

    R = -1
    n_examples_processed = 0
    cur_epoch_track = 0
    compute_R_train_error_time = 0
    train_error_time = 0
    loss_value = -1

    while not sv.should_stop():
      cur_iteration += 1
      sys.stdout.flush()

      start_time = time.time()

      # Compute batchsize ratio
      new_epoch_float = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
      new_epoch_track = int(new_epoch_float)
      cur_epoch_track = max(cur_epoch_track, new_epoch_track)

      run_options = tf.RunOptions()
      run_metadata = tf.RunMetadata()

      if FLAGS.timeline_logging:
        run_options.trace_level=tf.RunOptions.FULL_TRACE
        run_options.output_partition_graphs=True

      # Dequeue variable batchsize inputs
      images_real, labels_real = sess.run(variable_batchsize_inputs[FLAGS.batch_size])
      loss_value, step = sess.run([train_op, model.global_step], run_metadata=run_metadata, options=run_options, feed_dict={images:images_real, labels:labels_real})
      n_examples_processed += FLAGS.batch_size * num_workers

      # This uses the queuerunner which does not support variable batch sizes
      #loss_value, step = sess.run([train_op, global_step], run_metadata=run_metadata, options=run_options)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # Log the elapsed time per iteration
      finish_time = time.time()

      # Create the Timeline object, and write it to a json
      if FLAGS.timeline_logging:
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('%s/worker=%d_timeline_iter=%d.json' % (FLAGS.train_dir, FLAGS.task_id, step), 'w') as f:
          f.write(ctf)

      if step > FLAGS.max_steps:
        break

      cur_epoch = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
      tf.logging.info("epoch: %f time %f" % (cur_epoch, time.time()-begin_time));
      if cur_epoch >= FLAGS.n_train_epochs:
        break

      duration = time.time() - start_time
      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('Worker %d: %s: step %d, loss = %f'
                    '(%.1f examples/sec; %.3f  sec/batch)')
      tf.logging.info(format_str %
                      (FLAGS.task_id, datetime.now(), step, loss_value,
                       examples_per_sec, duration))

      # Determine if the summary_op should be run on the chief worker.
      if is_chief and next_summary_time < time.time() and FLAGS.should_summarize:

        tf.logging.info('Running Summary operation on the chief.')
        summary_str = sess.run(summary_op)
        sv.summary_computed(sess, summary_str)
        tf.logging.info('Finished running Summary operation.')

        # Determine the next time for running the summary.
        next_summary_time += FLAGS.save_summaries_secs

    if is_chief:
      tf.logging.info('Elapsed Time: %f' % (time.time()-begin_time))

    # Stop the supervisor.  This also waits for service threads to finish.
    sv.stop()

    # Save after the training ends.
    if is_chief:
      saver.save(sess,
                 os.path.join(FLAGS.train_dir, 'model.ckpt'),
                 global_step=model.global_step)
