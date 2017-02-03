import matplotlib.pyplot as plt
import sys
import re

def get_histogram_times(f):
    f = open(fname)
    dequeue_times = {}
    enqueue_times = {}
    for line in f:
        dequeue_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) ([0-9]+) Dequeueing\[([0-9]+)\]"
        enqueue_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) Enqueueing\[([0-9]+)\]"
        dequeue_m = re.match(dequeue_pattern, line)
        enqueue_m = re.match(enqueue_pattern, line)
        if dequeue_m:
            time_ms = dequeue_mgroup(1)
            worker = dequeue_mgroup(2)
            iteration = dequeue_mgroup(3)
            if iteration not in iteration_times:
                iteration_times[iteration] = []
            dequeue_times[iteration].append(time_ms)
        if enqueue_m:
            time_ms = enqueue_m.group(1)
            iteration = enqueue_m.group(2)
            enqueue_times[iteration] = time_ms
    f.close()

fname = sys.argv[1]
times = get_histogram_times(fname)
plt.hist(times)
plt.title(fname)
plt.savefig(fname + ".png")
