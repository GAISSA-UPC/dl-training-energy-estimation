import itertools
from pathlib import Path
from typing import List

import numpy as np

from src.data.datasets import Datasets
from src.environment import OUTPUT_DIR, ROOT, SEED
from src.models.dl.model_factory import Architectures
from src.profiling import REPETITIONS
from src.utils import read_config

random_generator = np.random.default_rng(seed=SEED)


def generate_variables_combinations(
    to_file: Path = None,
    exclude_architectures: List[str] = None,
    exclude_datasets: List[str] = None,
    input_sizes: List[str] = None,
    batch_sizes: List[str] = None,
):
    """
    Generates all possible combinations of architectures and datasets and repeats them REPETITIONS times and then
    shuffles them.

    Parameters
    ----------
    to_file: Path
        The path to the file where the combinations will be written.
    exclude_architectures: List[str]
        The list of architectures to exclude.
    exclude_datasets: List[str]
        The list of datasets to exclude.
    input_sizes: List[str]
        The list of input sizes.
    batch_sizes: List[str]
        The list of batch sizes.

    Returns
    -------
    generator: tuple
        The generator of all possible combinations of architectures and datasets.
    """

    if not isinstance(exclude_architectures, list):
        exclude_architectures = []
    if not isinstance(exclude_datasets, list):
        exclude_datasets = []

    architectures = [arch.value for arch in Architectures if arch.value not in exclude_architectures]
    datasets = [dataset.value for dataset in Datasets if dataset.value not in exclude_datasets]

    # Create an iterator of all possible combinations of architectures and datasets and repeat it REPETITIONS times and
    # then shuffle it
    variables_product = itertools.product(architectures, datasets, input_sizes, batch_sizes)
    experiment_variables = itertools.chain(*list(itertools.repeat(list(variables_product), REPETITIONS)))
    experiment_variables = list(experiment_variables)
    random_generator.shuffle(experiment_variables)

    if to_file is not None:
        _write_variables_combinations_to_file(to_file, experiment_variables)
    return _return_generator(experiment_variables)


def _write_variables_combinations_to_file(to_file: Path, experiment_variables: List[tuple]):
    with open(to_file, "w", encoding="utf8") as f:
        f.write("architecture,dataset,input_size,batch_size\n")
        for variables in experiment_variables:
            f.write(f"{','.join(variables)}\n")


def _return_generator(experiment_variables: List[tuple]) -> tuple:
    for variable in experiment_variables:
        yield variable


if __name__ == "__main__":
    # config = read_config(ROOT / "config" / "experiment_2.yaml")
    # exclude_architectures = config["EXCLUDE_ARCHITECTURES"]
    # exclude_datasets = config["EXCLUDE_DATASETS"]
    # input_sizes = [str(size) for size in config["INPUT_SIZES"]]
    # batch_sizes = [str(size) for size in config["BATCH_SIZES"]]

    # generate_variables_combinations(
    # to_file=OUTPUT_DIR / config["VARIABLES_COMBINATIONS"],
    #     exclude_architectures=exclude_architectures,
    #     exclude_datasets=exclude_datasets,
    #     input_sizes=input_sizes,
    #     batch_sizes=batch_sizes,
    # )

    config = read_config(ROOT / "config" / "experiment_2.yaml")
    exclude_architectures = config["EXCLUDE_ARCHITECTURES"]
    exclude_datasets = config["EXCLUDE_DATASETS"]
    input_sizes = [str(size) for size in config["INPUT_SIZES"]]
    batch_sizes = [str(size) for size in config["BATCH_SIZES"]]

    generate_variables_combinations(
        to_file=OUTPUT_DIR / config["VARIABLES_COMBINATIONS"],
        exclude_architectures=exclude_architectures,
        exclude_datasets=exclude_datasets,
        input_sizes=input_sizes,
        batch_sizes=batch_sizes,
    )
