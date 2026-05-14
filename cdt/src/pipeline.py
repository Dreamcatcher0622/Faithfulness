import os
import json
import argparse
from subprocess import Popen
import subprocess
from threading import Timer
import shutil
import time
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('--full_data_path', type=str, default="../data/input_data/Qwen3-1.7B/last_turn/expel.json")
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--only_ig", type = str, default = "True")
    args = parser.parse_args()


    full_data_path = args.full_data_path
    full_data= auto_read_data(full_data_path)
    for tid in range(len(full_data)):
        if tid < args.start or tid >= args.end:
            continue
        process = Popen([
            "python", "test_score_expel.py",
            "--task_id", f"{tid}",
            "--start_layer", f"{args.start_layer}",
            "--full_data_path", f"{full_data_path}",
            "--model_path", f"{args.model_path}",
            "--save_dir", "../results",
            "--only_ig", f"{args.only_ig}"
        ])
        process.wait()