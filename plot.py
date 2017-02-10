import matplotlib.pyplot as plt
import sys
import re

def get_histogram_times(f):
    f = open(fname)
    finished_times = {}
    start_times = {}
    max_iter = 0
    for line in f:
        dequeue_pattern = "INFO:tensorflow:Dequeue time - ([0-9]+) ([0-9.]+)"
        finish_pattern = "INFO:tensorflow:Finish time - ([0-9]+) ([0-9.]+)"

        dequeue_m = re.match(dequeue_pattern, line)
        finished_m = re.match(finish_pattern, line)

        if finished_m:
            step = int(finished_m.group(1))
            time_ms = float(finished_m.group(2))
            finished_times[step] = time_ms
            max_iter = max(max_iter, step)
        if dequeue_m:
            step = int(dequeue_m.group(1))
            time_ms = float(dequeue_m.group(2))
            start_times[step] = time_ms

    times = []
    for iteration in range(5, max_iter):
        if iteration in start_times and iteration in finished_times:
            if start_times[iteration] < finished_times[iteration]:
                times.append(finished_times[iteration]-start_times[iteration])
    f.close()
    print(sorted(times))
    return sorted(times)

fname = sys.argv[1]
times = get_histogram_times(fname)
plt.hist(times, bins=100)
plt.title(fname)
plt.savefig(fname + ".png")
