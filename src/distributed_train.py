# This code is taken and modified from the inception_distribute_train.py file of
# google's tensorflow inception model. The original source is here - https://github.com/tensorflow/models/tree/master/inception.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from threading import Timer
from sync_replicas_optimizer_modified.sync_replicas_optimizer_modified import TimeoutReplicasOptimizer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
import mnist

from timeout_manager import launch_manager

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('timeout_method', True, 'Use the timeout straggler killing method')
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

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for timeout communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 20,
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
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def train(target, dataset, cluster_spec):

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
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples / FLAGS.batch_size)

    # Decay steps need to be divided by the number of replicas to aggregate.
    # This was the old decay schedule. Don't want this since it decays too fast with a fixed learning rate.
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay / num_replicas_to_aggregate)
    # New decay schedule. Decay every few steps.
    #decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    images, labels = mnist.placeholder_inputs(FLAGS.batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    logits, reg = mnist.inference(images, train=True)

    # Add classification loss.
    total_loss = mnist.loss(logits, labels) + reg

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(lr)

    # Use V2 optimizer
    if not FLAGS.timeout_method:
      opt = tf.train.SyncReplicasOptimizerV2(
        opt,
        replicas_to_aggregate=num_replicas_to_aggregate,
        total_num_replicas=num_workers)
    else:
      opt = TimeoutReplicasOptimizer(
        opt,
        global_step,
        total_num_replicas=num_workers)

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(total_loss)
    if not FLAGS.timeout_method:
      apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
    else:
      apply_gradients_op = opt.apply_gradients(grads, FLAGS.task_id, global_step=global_step)
      timeout_op = opt.timeout_op
      wait_op = opt.wait_op

    with tf.control_dependencies([apply_gradients_op]):
      train_op = tf.identity(total_loss, name='train_op')

    a = data_flow_ops.FIFOQueue(-1, tf.float32)
    b = a.dequeue()

    # Get chief queue_runners, init_tokens and clean_up_op, which is used to
    # synchronize replicas.
    # More details can be found in sync_replicas_optimizer.
    chief_queue_runners = [opt.get_chief_queue_runner()]
    init_tokens_op = opt.get_init_tokens_op()
    #clean_up_op = opt.get_clean_up_op()

    # Create a saver.
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init_op = tf.initialize_all_variables()

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
                             global_step=global_step,
                             saver=saver,
                             save_model_secs=FLAGS.save_interval_secs)


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

    # TIMEOUT client overseer
    if FLAGS.timeout_method:
      timeout_client, timeout_server = launch_manager(sess, timeout_op, FLAGS)

    # Train, checking for Nans. Concurrently run the summary operation at a
    # specified interval. Note that the summary_op and train_op never run
    # simultaneously in order to prevent running out of GPU memory.
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    begin_time = time.time()
    cur_iteration = -1
    iterations_finished = set()

    def print_queue_sizes():
      tf.logging.info("Periodic print queue sizes...")
      sess.run([opt.print_sizes])
      sess.run([opt.print_p1_sizes])
      sess.run([opt.print_accum_sizes])
      sess.run([opt.print_local_step])
      tf.logging.info("Done periodic print queue sizes...")
      Timer(20, print_queue_sizes).start()
    Timer(10, print_queue_sizes).start()

    while not sv.should_stop():
      try:

        tf.logging.info("EXCEPTIONS QUEUERUNNER")
        tf.logging.info(len(opt._chief_queue_runner.exceptions_raiwed()))
        tf.logging.info(opt._chief_queue_runner.exceptions_raised())

        sys.stdout.flush()
        tf.logging.info("A new iteration...")

        # Increment current iteration
        cur_iteration += 1

        # Timeout method
        if FLAGS.timeout_method:

          # Broadcast worker starting iteration to other workers.
          timeout_client.broadcast_worker_starting(cur_iteration)

        # Wait for the queue to have a token before starting.
        sess.run([wait_op])

        #tf.logging.info("Printing sizes...")

        #print_queue_sizes()

        #tf.logging.info("Done printing sizes...")

        #sess.run([opt.print_sizes])

        #assert(cur_iteration == int(sess.run(global_step)))

        # Broadcast the iteration has begun.
        timeout_server.notify_iteration_starting(cur_iteration)

        start_time = time.time()
        feed_dict = mnist.fill_feed_dict(dataset, images, labels, FLAGS.batch_size)

        if FLAGS.timeline_logging:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          loss_value, step = sess.run([train_op, global_step], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
        else:
          if timeout_server.timeout < 0:
            loss_value, step = sess.run([train_op, global_step], feed_dict=feed_dict)
          else:
            tf.logging.info("Setting timeout: %d ms" % timeout_server.timeout)
            run_options = tf.RunOptions(timeout_in_ms=timeout_server.timeout)
            loss_value, step = sess.run([train_op, global_step], feed_dict=feed_dict, options=run_options)
            #Timer(timeout_server.timeout / float(1000), lambda : sess.run([timeout_op])).start()
            #loss_value, step = sess.run([train_op, global_step], feed_dict=feed_dict)

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
      except tf.errors.DeadlineExceededError:
        tf.logging.info("Timeout exceeded, running timeout op on iteration %d - %f" % (cur_iteration, time.time()))
        sess.run([timeout_op])
        tf.logging.info("Done executing timeout op")
      except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    if is_chief:
      tf.logging.info('Elapsed Time: %f' % (time.time()-begin_time))

    # Stop the supervisor.  This also waits for service threads to finish.
    sv.stop()

    # Save after the training ends.
    if is_chief:
      saver.save(sess,
                 os.path.join(FLAGS.train_dir, 'model.ckpt'),
                 global_step=global_step)
