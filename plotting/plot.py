import sys
import matplotlib.pyplot as plt
import numpy as np

"""
Cifar10 # of steps to 95% train error
batchsize, number of steps
128 - [6264, 5752, 5712]
256 - [3264, 3168, 3280]
512 - [2859, 2919, 2820]

* batch_size = # of gradients computed by all workers per step
batch_size num_workers epoch_reached, time
128 8 3.778560 411.244766
256 8 3.870720 219.264533
512 8 4.044800 118.744260
128 4 4.270080 265.423899
256 4 3.886080 127.269140
512 4 8.980480 121.987429
128 16 2.065920 470.391963
256 16 1.904640 207.247085
512 16 2.211840 120.478786
"""

steps_to_95_accuracy = {
    64 : [18000],
    128 : [6264, 5752, 5712],
    256 : [3264, 3168, 3280],
    512 : [2859, 2919, 2820],
    1024 : [2516],
    2048 : [2275]
}

train_data_size = 60000
n_workers=8
steps_to_95_accuracy_avg = [(k, sum(v) / float(len(v))) for k,v in steps_to_95_accuracy.items()]
steps_to_95_accuracy_avg.sort(key=lambda x:x[0])
accuracy_xs = [x[0] for x in steps_to_95_accuracy_avg]
accuracy_ys = [x[1] * n_workers / float(train_data_size) for x in steps_to_95_accuracy_avg]
plt.plot(accuracy_xs, accuracy_ys, label="95% train accuracy, g2.2xlarge", marker="o")
plt.title("Cifar10 - number of epochs to 95% train accuracy")
plt.ylabel("Number of Epochs")
plt.xlabel("Batchsize")
plt.legend(loc="upper right")
plt.savefig("Cifar10Accuracy_BatchsizeVsSteps")

###################################################
batchsize_workers_epoch_time = """
128 8 3.778560 411.244766
256 8 3.870720 219.264533
512 8 4.044800 118.744260
128 4 4.270080 265.423899
256 4 3.886080 127.269140
512 4 8.980480 121.987429
128 16 2.065920 470.391963
256 16 1.904640 207.247085
512 16 2.211840 120.478786
"""
plt.cla()
bwet = []
by_batchsize = {}
for line in batchsize_workers_epoch_time.splitlines():
    if line != "":
        bs, workers, epoch, time = tuple(float(x) for x in line.split())
        bwet.append((bs, workers, epoch, time))
        if bs not in by_batchsize:
            by_batchsize[bs] = []
        by_batchsize[bs].append((workers, time / float(epoch)))

for batchsize, vals in by_batchsize.items():
    vals.sort(key=lambda x:x[0])
    ys = [val[1] for val in vals]
    xs = [val[0] for val in vals]
    plt.plot(xs, ys, label="g2.2 Batchsize=%d"%int(batchsize), marker="o")

plt.ylabel("Time for one pass of data")
plt.xlabel("Number of workers")
plt.title("Cifar10 Epoch Time Speedup")
plt.legend(loc="upper left")
plt.savefig("Cifar10EpochTimeWorkerSpeedup.png")
print(bwet)
