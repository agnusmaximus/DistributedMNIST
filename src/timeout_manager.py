import signal
import sys
import os

from twisted.spread import pb
from twisted.internet import reactor
from threading import Thread, Timer

##################
# RPC procedures #
##################
class TimeoutServer(pb.Root):
  def __init__(self):
    self.worker_id = FLAGS.task_id
    self.n_total_workers = len(FLAGS.worker_hosts.split(","))
    self.iteration_track = [0] * self.n_total_workers
    self.n_to_collect = FLAGS.num_replicas_to_aggregate
    self.ready_to_start = False
    self.iterations_killed = set()
    tf.logging.info("Worker %d: starting status server..." % FLAGS.task_id)

    # When to collect statistico
    self.iteration_start_collect = 10
    self.iteration_end_collect = 80

    # Statistics tracking
    self.worker_start_times = {}
    for i in range(self.n_total_workers):
      self.worker_start_times[i] = {}

    # When to timeout
    self.timeout = -1

  def should_collect_statistics(self, iteration):
    return iteration > self.iteration_start_collect and iteration < self.iteration_end_collect

  # Keep track of statistics of iterations start times
  def track_statistics(self, worker_id, iteration, time):
    if self.should_collect_statistics(iteration):
      # Track worker start times.
      # Worker start time is defined to be the time at which a worker can start
      # computing gradients. (Note this is not the same as iteration start time,
      # which is when the parameters have been updated.)
      self.worker_start_times[worker_id][iteration] = time


  # Called when worker_id notifies this machine that it is starting iteration.
  def remote_notify_starting(self, worker_id, iteration):
    cur_time = time.time()
    tf.logging.info("Worker %d: Was notified that worker %d started iteration %d - t=%f" % (self.worker_id, worker_id, iteration, cur_time))
    self.iteration_track[worker_id] = iteration
    self.track_statistics(worker_id, iteration, cur_time)

  def remote_notify_ready_to_start(self):
    tf.logging.info("Server ready to start!")
    self.ready_to_start = True

  def remote_is_ready_to_start(self):
    return (self.worker_id, self.ready_to_start)

class TimeoutClient:
  def __init__(self):
    self.worker_id = FLAGS.task_id
    hosts = FLAGS.worker_hosts.split(",")
    hosts = [x.split(":")[0] for x in hosts]
    self.hosts = hosts
    self.self_perspective = None
    self.perspectives = []
    self.ready = False
    self.servers_ready = set([])

    for i, host in enumerate(hosts):
      factory = pb.PBClientFactory()
      tf.logging.info("Connecting to %s:%d" % (host, FLAGS.rpc_port))
      reactor.connectTCP(host, FLAGS.rpc_port, factory)
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

  def broadcast_starting(self, iteration):
    for persp in self.perspectives:
      persp.callRemote("notify_starting", self.worker_id, iteration).addCallbacks(self.success, self.fail)

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
    tf.logging.info("Trying reconnecting to %s:%d" % (host, FLAGS.rpc_port))
    reactor.connectTCP(host, FLAGS.rpc_port, factory)
    factory.getRootObject().addCallbacks(self.connected, self.connect_failure, errbackArgs=(host))

# Separate manager process to oversee training on workers.
def launch_manager(sess, timeout_op):
  # Launch a separate thread in the background that checks whether the
  # machine is a straggler.
  timeout_server = TimeoutServer()
  rpc_server = pb.PBServerFactory(timeout_server)
  reactor.listenTCP(FLAGS.rpc_port, rpc_server)
  rpc_client = TimeoutClient()
  Thread(target=reactor.run, args=(False,)).start()

  while not rpc_client.ready_to_start():
    rpc_client.check_ready_to_start()
    time.sleep(1)

  return rpc_client, timeout_server,
