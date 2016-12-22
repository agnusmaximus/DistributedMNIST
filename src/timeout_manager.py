import signal
import sys
import os

import tensorflow as tf
from twisted.spread import pb
from twisted.internet import reactor
from threading import Thread, Timer

##################
# RPC procedures #
##################
class TimeoutServer(pb.Root):
  def __init__(self, tf_flags):
    self.tf_flags = tf_flags
    self.worker_id = self.tf_flags.task_id
    self.n_total_workers = len(self.tf_flags.worker_hosts.split(","))
    self.iteration_track = [0] * self.n_total_workers
    self.n_to_collect = self.tf_flags.num_replicas_to_aggregate
    self.ready_to_start = False
    self.iterations_killed = set()
    tf.logging.info("Worker %d: starting status server..." % self.tf_flags.task_id)

    # Statistics tracking
    self.worker_start_times = {}
    for i in range(self.n_total_workers):
      self.worker_start_times[i] = {}
    self.iteration_start_times = {}

    # When to timeout
    self.timeout = -1

  # Keep track of statistics of iterations start times
  def track_worker_start_times(self, worker_id, iteration, time):
    # Track worker start times.
    # Worker start time is defined to be the time at which a worker can start
    # computing gradients. (Note this is not the same as iteration start time,
    # which is when the parameters have been updated.)
    self.worker_start_times[worker_id][iteration] = time

  def track_iteration_start_times(self, iteration, time):
    self.iteration_start_times[iteration] = time

  # Called when worker_id notifies this machine that it is starting iteration.
  def remote_notify_worker_starting(self, worker_id, iteration):
    cur_time = time.time()
    tf.logging.info("Worker %d: Was notified that worker %d started iteration %d - t=%f" % (self.worker_id, worker_id, iteration, cur_time))
    self.iteration_track[worker_id] = iteration
    self.track_worker_start_times(worker_id, iteration, cur_time)

  def notify_iteration_starting(self, iteration):
    cur_time = time.time()
    tf.logging.info("Beginning of iteration %d (dequeued successfully)" % iteration)
    self.track_iteration_start_times(iteration, cur_time)

  def remote_notify_ready_to_start(self):
    tf.logging.info("Server ready to start!")
    self.ready_to_start = True

  def remote_is_ready_to_start(self):
    return (self.worker_id, self.ready_to_start)

class TimeoutClient:
  def __init__(self, tf_flags):
    self._tf_flags = tf_flags
    self.worker_id = self.tf_flags.task_id
    hosts = self.tf_flags.worker_hosts.split(",")
    hosts = [x.split(":")[0] for x in hosts]
    self.hosts = hosts
    self.self_perspective = None
    self.perspectives = []
    self.ready = False
    self.servers_ready = set([])

    for i, host in enumerate(hosts):
      factory = pb.PBClientFactory()
      tf.logging.info("Connecting to %s:%d" % (host, self.tf_flags.rpc_port))
      reactor.connectTCP(host, self.tf_flags.rpc_port, factory)
      if i == self.worker_id:
        factory.getRootObject().addCallbacks(self.connected_self, self.connect_failure, errbackArgs=[host], errbackKeywords=[])
      else:
        factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=[host], errbackKeywords=[])

  def server_ready_to_start(self, *args):
    wid, ready = args[0]
    if ready:
      tf.logging.info("Worker %d is ready to begin..." % wid)
      self.servers_ready.add(wid)

  def check_ready_to_start(self):
    for persp in self.perspectives:
      persp.callRemote("is_ready_to_start").addCallbacks(self.server_ready_to_start, self.fail)

  def ready_to_start(self):
    return self.ready and len(self.servers_ready) == len(self.hosts)

  def signal_server_ready(self):
    tf.logging.info("Signaling ready to self's server")
    self.self_perspective.callRemote("notify_ready_to_start").addCallbacks(self.success, self.fail)

  def broadcast_worker_starting(self, iteration):
    for persp in self.perspectives:
      persp.callRemote("notify_worker_starting", self.worker_id, iteration).addCallbacks(self.success, self.fail)

  def connected(self, perspective):
    self.perspectives.append(perspective)
    tf.logging.info("Connected!")
    self.ready = (len(self.hosts) == len(self.perspectives))
    if self.ready:
      tf.logging.info("Ready!")
      self.signal_server_ready()
    else:
      tf.logging.info("%d of %d" % (len(self.perspectives), len(self.hosts)))

  def connected_self(self, perspective):
    self.self_perspective = perspective
    self.connected(perspective)

  def success(self, result):
    #tf.logging.info("Success!")
    pass

  def fail(self, _):
    tf.logging.info("Fail")
    tf.logging.info(_)

  def connect_failure(self, *args, **kwargs):
    tf.logging.info("RPC error, something failed: ")
    time.sleep(1)
    host = "".join(args[1:])
    factory = pb.PBClientFactory()
    tf.logging.info("Trying reconnecting to %s:%d" % (host, self.tf_flags.rpc_port))
    reactor.connectTCP(host, self.tf_flags.rpc_port, factory)
    factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=(host))

# Separate manager process to oversee training on workers.
def launch_manager(sess, timeout_op, tf_flags):
  # Launch a separate thread in the background that checks whether the
  # machine is a straggler.
  timeout_server = TimeoutServer(tf_flags)
  rpc_server = pb.PBServerFactory(timeout_server)
  reactor.listenTCP(tf_flags.rpc_port, rpc_server)
  rpc_client = TimeoutClient(tf_flags)
  Thread(target=reactor.run, args=(False,)).start()

  while not rpc_client.ready_to_start():
    rpc_client.check_ready_to_start()
    time.sleep(1)

  return rpc_client, timeout_server,
