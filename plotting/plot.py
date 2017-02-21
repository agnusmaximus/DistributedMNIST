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

Cifar10 # of steps to 95% train error
batchsize, number of steps
64 - [26210] # lr = .8
128 - [6045, 6166] # lr = .8
256 - [4094, 3488, 3707] # lr = .1
512 - [2756, 3084, 2618] # lr = .1
1024 - [2946, 2616, 2462] # lr = .1
2048 - [2452, 2403, 2447] # lr = .1
"""

# Old steps to 95 accuracy data
"""64 : [18000],
128 : [6264, 5752, 5712],
256 : [3264, 3168, 3280],
512 : [2859, 2919, 2820],
1024 : [2516],
2048 : [2275]"""

steps_to_95_accuracy = {
    64 : [26210], # lr = .8
    128 : [6045, 6166], # lr = .8
    256 : [4094, 3488, 3707], # lr = .1
    512 : [2756, 3084, 2618], # lr = .1
    1024 : [2946, 2616, 2462], # lr = .1
    2048 : [2452, 2403, 2447], # lr = .1
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

workers = sorted(list(set([b[1] for b in bwet])))
plt.plot(workers, workers, label="Optimal speedup")

plt.ylabel("Speedup over one worker per epoch")
plt.xlabel("Number of workers")
plt.title("Cifar10 Epoch Time Speedup")
plt.legend(loc="upper left")
plt.savefig("Cifar10EpochTimeWorkerSpeedup.png")
print(bwet)


ratios ="""
Step: 0
Ratio: 2678.689088
2017-02-14 11:38:07.483631: precision @ 1 = 0.109
Step: 139
Ratio: 2041.887889
2017-02-14 11:39:49.030979: precision @ 1 = 0.357
Step: 497
Ratio: 2070.008372
2017-02-14 11:41:30.732961: precision @ 1 = 0.500
Step: 856
Ratio: 2132.638985
2017-02-14 11:43:12.753396: precision @ 1 = 0.606
Step: 1250
Ratio: 2240.432371
2017-02-14 11:44:54.326838: precision @ 1 = 0.678
Step: 1608
Ratio: 2121.988342
2017-02-14 11:46:36.024351: precision @ 1 = 0.672
Step: 1967
Ratio: 2149.175142
2017-02-14 11:48:17.857121: precision @ 1 = 0.733
Step: 2325
Ratio: 2226.469264
2017-02-14 11:49:59.947401: precision @ 1 = 0.777
Step: 2683
Ratio: 2499.374460
2017-02-14 11:51:42.714091: precision @ 1 = 0.836
Step: 3078
Ratio: 2189.867502
2017-02-14 11:53:25.897244: precision @ 1 = 0.846
Step: 3436
Ratio: 3141.760238
2017-02-14 11:55:07.527295: precision @ 1 = 0.911
Step: 3794
Ratio: 2631.409370
2017-02-14 11:56:49.215963: precision @ 1 = 0.933
Step: 4153
Ratio: 2523.594715
2017-02-14 11:58:31.097979: precision @ 1 = 0.961
Step: 4511
Ratio: 2174.401953
2017-02-14 12:00:12.916297: precision @ 1 = 0.940
Step: 4905
Ratio: 2949.337159
2017-02-14 12:01:54.735326: precision @ 1 = 0.993
Step: 5264
Ratio: 4355.481349
2017-02-14 12:03:36.588436: precision @ 1 = 0.998
Step: 5622
Ratio: 2221.645186
2017-02-14 12:05:18.152155: precision @ 1 = 0.907
Step: 5980
Ratio: 2442.003448
2017-02-14 12:06:59.914687: precision @ 1 = 0.955
Step: 6339
Ratio: 5669.054728
2017-02-14 12:08:41.648752: precision @ 1 = 0.999
Step: 6697
Ratio: 5573.187830
2017-02-14 12:10:23.890670: precision @ 1 = 0.999
Step: 7091
"""
plt.cla()
precisions, xs, ys = [], [], []
for line in ratios.splitlines():
    if line.strip() == "":
        continue
    if "precision @ 1" in line:
        precision = float(line.split(" ")[-1])
        precisions.append(precision)
    else:
        name, value = line.split(":")
        if name == "Step":
            xs.append(float(value))
        elif name == "Ratio":
            ys.append(float(value))

min_length = min(len(xs), len(ys), len(precisions))
precisions = precisions[:min_length]
xs = xs[:min_length]
epochs = [x * 8 * 128 / 60000 for x in xs]
ys = ys[:min_length]

precisions_and_ratios = [(precisions[i], ys[i]) for i in range(min_length)]
precisions_and_ratios.sort(key=lambda x : x[0])
xs_ratio = [x[0] for x in precisions_and_ratios]
ys_ratio = [x[1] for x in precisions_and_ratios]

plt.plot(xs, ys)
plt.xlabel("Step")
plt.ylabel("Batchsize Ratio")
plt.title("Step vs Batchsize Ratio")
plt.savefig("StepVsBatchsizeRatio.png")

plt.cla()
plt.plot(epochs, ys)
plt.xlabel("Epoch")
plt.ylabel("Batchsize Ratio")
plt.title("Epoch vs Batchsize Ratio")
plt.savefig("EpochVsBatchsizeRatio.png")

plt.cla()
plt.plot(xs_ratio, ys_ratio)
plt.xlabel("Training Accuracy")
plt.ylabel("Batchsize Ratio")
plt.title("Training accuracy vs Batchsize Ratio")
plt.savefig("TrainingAccuracyVsBatchsizeRatio.png")

plt.cla()
plt.plot(xs, precisions)
plt.xlabel("Step")
plt.ylabel("Training accuracy")
plt.title("Step vs Training accuracy")
plt.savefig("TrainingAccuracy.png")
