import os
import time
from argparse import ArgumentParser

from src.data.datasets import Datasets
from src.environment import DATASET_DIR
from src.generate_combinations import generate_variables_combinations
from src.models.dl.model_factory import Architectures
from src.profiling import (
    COOLDOWN,
    EXCLUDE_ARCHITECTURES,
    EXCLUDE_DATASETS,
    MINUTES_TO_SECONDS,
)
from src.profiling.profile_models import profile_model, warmup


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "environment",
        help="The type of training environment.",
        choices=["local", "cloud"],
        type=str,
    )
    parser.add_argument(
        "config",
        help="The configuration file to use.",
        choices=sorted(
            [
                filename
                for filename in os.listdir("config")
                if filename.startswith("experiment") and filename.endswith(".yaml")
            ]
        ),
        type=str,
    )
    parser.add_argument("--warmup", help="Warmup the GPU.", action="store_true")

    subparsers = parser.add_subparsers(dest="single_run", help="sub-command help")

    parser_single_training = subparsers.add_parser("single-run", help="Single training help")
    parser_single_training.add_argument(
        "architecture",
        help="The architecture of the DNN.",
        choices=Architectures.to_list(),
        type=str,
    )
    parser_single_training.add_argument(
        "dataset", help="The dataset to use for training.", choices=Datasets.to_list(), type=str
    )
    parser_single_training.add_argument(
        "input_size",
        help="The input size of the images. Example: 224",
        type=int,
    )
    parser_single_training.add_argument(
        "batch_size",
        help="The batch size to use for training.",
        type=int,
    )
    parser_single_training.add_argument(
        "-d",
        "--data-path",
        help="Path to the dataset folder.",
        default=DATASET_DIR,
        type=str,
    )
    parser_single_training.add_argument(
        "--experiment-name",
        help="The name of the MLflow experiment.",
        type=str,
    )
    args = parser.parse_args()
    environment = args.environment.lower()
    os.environ["CONFIG_FILE"] = args.config
    single_run = args.single_run
    perform_warmup = args.warmup

    if single_run:
        architecture = args.architecture
        dataset = args.dataset
        input_size = args.input_size
        batch_size = args.batch_size
        data_folder = args.data_path
        experiment_name = args.experiment_name
        return environment, architecture, dataset, input_size, batch_size, data_folder, experiment_name, perform_warmup
    else:
        return environment, None, None, None, None, None, None, perform_warmup


if __name__ == "__main__":
    (
        environment,
        architecture,
        dataset,
        input_size,
        batch_size,
        data_folder,
        user_defined_experiment_name,
        perform_warmup,
    ) = parse_args()

    if perform_warmup:
        warmup()
        print(f"Waiting {COOLDOWN} minutes to cooldown before starting.")
        time.sleep(COOLDOWN * MINUTES_TO_SECONDS)

    if architecture and dataset:
        profile_model(
            environment, architecture, dataset, input_size, batch_size, user_defined_experiment_name, data_folder
        )
        exit(0)
    else:
        print("Start profiling...")

        experiment_generator = generate_variables_combinations(
            exclude_architectures=EXCLUDE_ARCHITECTURES, exclude_datasets=EXCLUDE_DATASETS
        )

        experiment_runs = dict()
        runs = 0
        for arch, dataset, input_size, batch_size in experiment_generator:
            experiment_run = experiment_runs.get((arch, dataset), 0)
            profile_model(
                environment,
                arch,
                dataset,
                input_size,
                batch_size,
                user_defined_experiment_name,
                data_folder,
                experiment_run,
                runs,
            )
            experiment_runs[(arch, dataset)] = experiment_run + 1
            runs += 1
        exit(0)
