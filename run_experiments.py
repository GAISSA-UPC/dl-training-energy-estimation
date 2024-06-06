#!/usr/bin/env python

import csv
import os
import shlex
import subprocess
import time
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd

from src.environment import CONFIGS_DIR, OUTPUT_DIR
from src.profiling import MINUTES_TO_SECONDS
from src.utils import read_config


def start_run(architecture, dataset, input_size, batch_size, experiment_name):
    run_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        process = subprocess.run(
            shlex.split(
                f"python3 runner.py {ENVIRONMENT} {CONFIG_FILE} single-run {architecture} {dataset} {input_size} {batch_size} --experiment-name {experiment_name}"
            ),
            check=True,
        )
        return_code = process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Process returned with code {e.returncode}")
        return_code = e.returncode
    csv_writer.writerow([run_start_time, architecture, dataset, input_size, batch_size, return_code])


parser = ArgumentParser()
parser.add_argument(
    "environment",
    help="The environment to run the profiling in.",
    choices=["local", "cloud"],
    type=str,
)
parser.add_argument(
    "config",
    help="The name of the configuration file to use.",
    choices=sorted(
        [
            filename
            for filename in os.listdir("config")
            if filename.startswith("experiment") and filename.endswith(".yaml")
        ]
    ),
    type=str,
)
parser.add_argument(
    "--experiment-name",
    help="The name of the MLflow experiment.",
    type=str,
)

args = parser.parse_args()
ENVIRONMENT = args.environment
CONFIG_FILE = args.config
experiment_name = args.experiment_name
os.environ["CONFIG_FILE"] = CONFIG_FILE

config = read_config(CONFIGS_DIR / CONFIG_FILE)
COOLDOWN = config["COOLDOWN"]
COOLDOWN_EVERY = config["COOLDOWN_EVERY"]


# Read the csv file in variables-combinations.csv
variables_combinations = pd.read_csv(
    OUTPUT_DIR / config["VARIABLES_COMBINATIONS"],
    header=0,
    dtype={"architecture": str, "dataset": str, "input_size": str, "batch_size": str},
)

# Run the warmup script
subprocess.run(shlex.split("python3 -c 'from src.profiling.profile_models import warmup; warmup()'"), check=False)
print(f"Waiting {COOLDOWN} minutes to cooldown after warmup.")
time.sleep(COOLDOWN * MINUTES_TO_SECONDS)

creation_time = datetime.now().strftime("%Y%m%dT%H%M%S")
with open(OUTPUT_DIR / f"oom-experiments_{creation_time}.csv", "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    csv_writer.writerow(["start_time", "architecture", "dataset", "input_size", "batch_size", "return_code"])

    # For each architecture and dataset combination, run the following command:
    # python runner.py --warmup local single-run <architecture> <dataset>
    for index, row in variables_combinations.iterrows():
        architecture = row["architecture"]
        dataset = row["dataset"]
        input_size = row["input_size"]
        batch_size = row["batch_size"]

        if COOLDOWN_EVERY > 0 and index != 0 and index % COOLDOWN_EVERY == 0:
            print(f"Waiting {COOLDOWN} minutes to cooldown")
            time.sleep(COOLDOWN * MINUTES_TO_SECONDS)

        if experiment_name is None:
            experiment_name = f"{ENVIRONMENT}-{dataset}-{architecture}"

        start_run(architecture, dataset, input_size, batch_size, experiment_name)
        f.flush()
