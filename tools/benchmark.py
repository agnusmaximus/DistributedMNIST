from __future__ import print_function
import sys
import time
from tf_ec2 import tf_ec2_run, Cfg

def load_cfg_from_file(cfg_file):
    cfg_f = open(cfg_file, "r")
    return eval(cfg_f.read())


def clean_run_and_download_evaluator_file(run_time_sec, cfg, evaluator_file_name="out_evaluator", outdir="result_dir"):

    shutdown_args = "tools/tf_ec2.py shutdown"
    tf_ec2_run(shutdown_args.split(), cfg)

    launch_and_run_args = "tools/tf_ec2.py clean_launch_and_run"
    cluster_specs = tf_ec2_run(launch_and_run_args.split(), cfg)
    cluster_string = cluster_specs["cluster_string"]

    time.sleep(run_time_sec)

    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, evaluator_file_name, outdir)
    tf_ec2_run(download_evaluator_file_args.split(), cfg)

def plot_time_loss(cfg1, cfg2, rerun=True):

    clean_run_and_download_evaluator_file(300, cfg1)
    #clean_run_and_download_evaluator_file(300, cfg2)
    pass

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python benchmark.py config_file1 config_file2")
        sys.exit(0)

    cfg_file1 = sys.argv[1]
    cfg_file2 = sys.argv[2]
    cfg1 = load_cfg_from_file(cfg_file1)
    cfg2 = load_cfg_from_file(cfg_file2)
    plot_time_loss(cfg1, cfg2)
