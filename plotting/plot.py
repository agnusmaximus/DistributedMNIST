import sys
import matplotlib.pyplot as plt
import numpy as np

"""
Cifar10 # of steps to 95% train error
batchsize, number of steps

64 - [18000+]
128 - [6264, 5752, 5712]
256 - [3264, 3168, 3280]
512 - [2859, 2919, 2820]
1024 - [2516]
2048 - [2275]

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

* batch_size_per_worker = # of gradients computed by a single worker per step
batch_size_per_worker num_workers epoch_reached time
64 1 2.636800 190.914938
128 1 2.534400 139.223831
256 1 3.886080 217.428507
512 1 3.645440 193.165316
4096 1 4.587520 238.769354
64 4 4.541440 154.375511
128 4 4.986880 76.427120
256 4 13.168640 171.589525
512 4 2.949120 37.487429
4096 4 16.384000 214.169686
64 8 6.553600 178.199938
128 8 8.437760 115.382619
256 8 18.145280 123.537439
512 8 9.912320 64.735797
4096 8 88.473600 563.110506
64 12 3.287040 107.078580
128 12 5.713920 77.975898
256 12 4.853760 33.834095
512 12 25.436160 105.502983
4096 12 22.609920 101.755663
64 16 8.560640 251.676567
512 16 20.807680 71.990144
4096 16 20.971520 64.395243
64 32 7.946240 224.600640
512 32 19.005440 64.613174
4096 32 83.886080 130.042110
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
n_workers = 8
steps_to_95_accuracy_avg = [(k, sum(v) / float(len(v))) for k,v in steps_to_95_accuracy.items()]
steps_to_95_accuracy_avg.sort(key=lambda x:x[0])
accuracy_xs = [x[0] for x in steps_to_95_accuracy_avg]
accuracy_ys = [x[1] * x[0] / float(train_data_size) for x in steps_to_95_accuracy_avg]
plt.plot(accuracy_xs, accuracy_ys, label="95% train accuracy, g2.2xlarge", marker="o")
plt.title("Cifar10 - number of epochs to 95% train accuracy")
plt.ylabel("Number of Epochs")
plt.xlabel("Batchsize")
plt.legend(loc="upper left")
plt.savefig("Cifar10Accuracy_BatchsizeVsSteps")

###################################################
batchsize_workers_epoch_time = """
64 1 2.636800 190.914938
512 1 3.645440 193.165316
4096 1 4.587520 238.769354
64 4 4.541440 154.375511
512 4 2.949120 37.487429
4096 4 16.384000 214.169686
64 8 6.553600 178.199938
512 8 9.912320 64.735797
4096 8 88.473600 563.110506
64 12 3.287040 107.078580
512 12 25.436160 105.502983
4096 12 22.609920 101.755663
64 16 8.560640 251.676567
512 16 20.807680 71.990144
4096 16 20.971520 64.395243
64 32 7.946240 224.600640
512 32 19.005440 64.613174
4096 32 83.886080 130.042110
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
    ys = [vals[0][1] / val[1] for val in vals if val[0] != 1]
    xs = [val[0] for val in vals if val[0] != 1]
    plt.plot(xs, ys, label="g2.2 worker_batchsize=%d"%int(batchsize), marker="o")

plt.plot(

plt.ylabel("Speedup over one worker per epoch")
plt.xlabel("Number of workers")
plt.title("Cifar10 Epoch Time Speedup")
plt.legend(loc="upper left")
plt.savefig("Cifar10EpochTimeWorkerSpeedup.png")
print(bwet)
