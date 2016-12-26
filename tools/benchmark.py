from __future__ import print_function
import sys
import time
import re
import shutil
import os
import glob
from matplotlib import pyplot as plt
from tf_ec2 import tf_ec2_run, Cfg

def load_cfg_from_file(cfg_file):
    cfg_f = open(cfg_file, "r")
    return eval(cfg_f.read())

def shutdown_and_launch(cfg):
    shutdown_args = "tools/tf_ec2.py shutdown"
    tf_ec2_run(shutdown_args.split(), cfg)

    launch_args = "tools/tf_ec2.py launch"
    tf_ec2_run(launch_args.split(), cfg)

def run_tf_and_download_evaluator_file(run_time_sec, cfg, evaluator_file_name="out_evaluator", outdir="result_dir"):

    kill_args = "tools/tf_ec2.py kill_all_python"
    tf_ec2_run(kill_args.split(), cfg)

    run_args = "tools/tf_ec2.py run_tf"
    cluster_specs = tf_ec2_run(run_args.split(), cfg)
    cluster_string = cluster_specs["cluster_string"]

    time.sleep(run_time_sec)

    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, evaluator_file_name, outdir)
    tf_ec2_run(download_evaluator_file_args.split(), cfg)

def extract_times_losses_precision(fname):
    f = open(fname)

    times, losses, precisions = [], [], []
    for line in f:
        m = re.match("Num examples: ([0-9]*)  Precision @ 1: ([\.0-9]*) Loss: ([\.0-9]*) Time: ([\.0-9]*)", line)
        if m:
            examples, precision, loss, time = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            times.append(time)
            losses.append(loss)
            precisions.append(precision)
    f.close()
    return times, losses, precisions

def plot_time_loss(cfgs, evaluator_file_name="out_evaluator", outdir="result_dir", time_limit=350, rerun=True, launch=True):

    shutil.rmtree(outdir)
    os.makedirs(outdir)

    if rerun:
        if launch:
            shutdown_and_launch(cfgs[0])
        for cfg in cfgs:
            run_tf_and_download_evaluator_file(time_limit, cfg, evaluator_file_name=evaluator_file_name, outdir=outdir)

    plt.xlabel("time (s)")
    plt.ylabel("loss")
    for fname in glob.glob(outdir + "/*"):
        label = fname.split("/")[-1]
        times, losses, precisions = extract_times_losses_precision(fname)
        print(times, losses, precisions)
        plt.plot(times, losses, linestyle='solid', label=label)
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig("time_loss.png")

if __name__ == "__main__":
    print("Usage: python benchmark.py config_file1 config_file2")
    cfgs = [str(x) for x in sys.argv[1:]]
    cfgs = [load_cfg_from_file(x) for x in cfgs]
    plot_time_loss(cfgs)
