import matplotlib.pyplot as plt
import sys
import re

def get_histogram_times(f):
    f = open(fname)
    finished_times = {}
    enqueue_times = {}
    max_iter = 0
    workers = set()
    for line in f:
        dequeue_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) ([0-9]+) Dequeueing\[([0-9]+)\]"
        enqueue_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) Enqueueing\[([0-9]+)\]"
        finished_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) ([0-9]+) Finished\[([0-9]+)\]"
        dequeue_m = re.match(dequeue_pattern, line)
        enqueue_m = re.match(enqueue_pattern, line)
        finished_m = re.match(finished_pattern, line)
        if finished_m:
            time_ms = int(finished_m.group(1))
            worker = int(finished_m.group(2))
            iteration = int(finished_m.group(3))
            if iteration not in finished_times:
                finished_times[iteration] = {}
            finished_times[iteration][worker] = time_ms
            max_iter = max(max_iter, iteration)
            workers.add(worker)
        if enqueue_m:
            time_ms = int(enqueue_m.group(1))
            iteration = int(enqueue_m.group(2))
            enqueue_times[iteration] = time_ms

    times = []
    for iteration in range(10, max_iter/2):
        if min([finished_times[iteration][k] for k in workers if k in finished_times[iteration]]) - enqueue_times[iteration] < 1000:
            print(enqueue_times[iteration])
            print([finished_times[iteration][k] for k in workers if k in finished_times[iteration]])
            print(sorted([finished_times[iteration][k]-enqueue_times[iteration] for k in workers if k in finished_times[iteration]]))
        for worker in workers:
            if iteration in finished_times and iteration in enqueue_times:
                if worker in finished_times[iteration]:
                    elapsed = finished_times[iteration][worker] - enqueue_times[iteration]
                    times.append(elapsed)
    f.close()
    return sorted(times)

fname = sys.argv[1]
times = get_histogram_times(fname)
plt.hist(times, bins=100)
plt.title(fname)
plt.savefig(fname + ".png")
