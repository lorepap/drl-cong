"""
Train DRL on multiple MahiMahi traces.
"""

import os
import yaml
import re
import sys
import subprocess
from argparse import ArgumentParser
from src.helper import context, utils

TRAINING_FILENAME = os.path.join(context.src_dir, "train.py")
# TESTING_FILENAME = os.path.join(context.ml_dir, "test.py")

def run_experiments():
    traces = ["tm.lte.driving", "vz.lte.driving"]
    for i, trace in enumerate(traces):
        print(f"Training PPO on {trace}")

        # if i > 0: # the same model will be retrained on multiple input traces
        #     retrain = 1
        # else:
        #     retrain = 0

        retrain = 1

        # generate command to execute for this trace        
        command = f"python3 {TRAINING_FILENAME} --trace {trace} --retrain {retrain}"
        print("Executing", command)
        # execute each command and wait for it to finish
        try:
            subprocess.check_call(command, shell=True, stderr=sys.stderr, stdin=sys.stdin, stdout=sys.stdout, bufsize=1)
        
        except subprocess.CalledProcessError as e:
            print(f"Error running command '{command}': {e}")
            os._exit(1)
    
    # # Testing the model
    # command = f"python3 {TESTING_FILENAME} -m {model} -t {trace} -x {ip}"
    # try:
    #     subprocess.check_call(command, shell=True, stderr=sys.stderr, stdin=sys.stdin, stdout=sys.stdout, bufsize=1)

    # except subprocess.CalledProcessError as e:
    #     print(f"Error running command '{command}': {e}")

    
if __name__ == "__main__":
    run_experiments()
