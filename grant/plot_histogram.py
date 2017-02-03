import matplotlib.pyplot as plt
import sys
import re

def get_histogram_times(f):
    f = open(fname)
    dequeue_times = {}
    enqueue_times = {}
    max_iter = 0
    workers = set()
    for line in f:
        dequeue_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) ([0-9]+) Dequeueing\[([0-9]+)\]"
        enqueue_pattern = "I tensorflow/core/kernels/logging_ops.cc:88] ([0-9]+) Enqueueing\[([0-9]+)\]"
        dequeue_m = re.match(dequeue_pattern, line)
        enqueue_m = re.match(enqueue_pattern, line)
        if dequeue_m:
            time_ms = int(dequeue_m.group(1))
            worker = int(dequeue_m.group(2))
            iteration = int(dequeue_m.group(3))
            if iteration not in dequeue_times:
                dequeue_times[iteration] = {}
            dequeue_times[iteration][worker] = time_ms
            max_iter = max(max_iter, iteration)
            workers.add(worker)
        if enqueue_m:
            time_ms = int(enqueue_m.group(1))
            iteration = int(enqueue_m.group(2))
            enqueue_times[iteration] = time_ms

    times = []
    for i in range(50, max_iter):
        for j in workers:
            if i in dequeue_times and i-1 in dequeue_times:
                if j in dequeue_times[i] and j in dequeue_times[i-1]:
                    times.append(dequeue_times[i][j]-dequeue_times[i-1][j])
    print(sorted(times))
    sys.exit(0)
    f.close()

fname = sys.argv[1]
times = get_histogram_times(fname)
plt.hist(times)
plt.title(fname)
plt.savefig(fname + ".png")
