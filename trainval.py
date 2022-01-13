import sys
from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_wizard as hw
from haven import haven_utils as hu
import hydra
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

# from src import models
# from src import datasets
# from src import utils as ut
from pretrain import main

import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from bunch import Bunch
from copy import deepcopy
cudnn.benchmark = True
import pickle as pkl


@hydra.main(config_path='.', config_name='pretrain')
def fun(cfg):
    global config
    config = cfg

def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    sys.argv=["pretrain.py", f"agent={exp_dict['agent_name']}", f"domain={exp_dict['domain']}"]
    conf = fun()
    # cfg.agent = Bunch(cfg.agent)
    config.snapshot_dir = exp_dict["snapshot_dir"]
    main(config, savedir)

    print("Experiment completed")


if __name__ == "__main__":
    import exp_configs
    import job_configs
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", "--exp_group_list", nargs="+", help="Define which exp groups to run."
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="",
        type=str,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-d", "--datadir", default=None, help="Define the dataset directory."
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument("-c", "--cuda", default=1, type=int)
    parser.add_argument("-j", "--job_scheduler", default=None, type=str)
    parser.add_argument("-p", "--python_binary_path", default='/mnt/home/urlbenv/bin/python')
    parser.add_argument("-nw", "--num_workers", default=0, type=int)
    args, others = parser.parse_known_args()

    if args.job_scheduler == "slurm":
        job_config = {
            "account_id": "rrg-bengioy-ad",
            "time": "12:00:00",
            "cpus-per-task": "2",
            "mem-per-cpu": "16G",
            "gres": "gpu:1",
        }
    elif args.job_scheduler == "toolkit":
        import job_configs
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    hw.run_wizard(
        func=trainval,
        exp_groups=exp_configs.EXP_GROUPS,
        job_config=job_configs.JOB_CONFIG,
        python_binary_path=args.python_binary_path,
        savedir_base=args.savedir_base,
        use_threads=True,
        args=args,
    )
