# dl-energy-estimation
Replication package for the paper "How to use model architecture and training environment to estimate the energy consumption of DL training".

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11505891.svg)](https://doi.org/10.5281/zenodo.11505891)

## Set up the environment
### Installing dependencies
Before executing the code, you must first install the required dependencies.
We use [Poetry](https://python-poetry.org/docs/) to manage the dependencies.

If you want to use any other dependency manager, you can look at the [pyproject.toml](pyproject.toml) file for the required dependencies.

### MLflow configuration
We use [MLflow](https://mlflow.org/docs/latest/index.html) to keep track of the different experiments. By default, its usage
is disabled. If you want to use MLflow, you need to:
- Configure your own [tracking server](https://mlflow.org/docs/latest/tracking.html#tracking-server).
- Activate MLflow logging in the corresponding [experiment_#.yaml](config/experiment_1.yaml) configuration file.
- Create a `.env` file in the project root with the following structure:
```text
MLFLOW_TRACKING_URI = https://url/to/your/tracking/server
MLFLOW_TRACKING_USERNAME = tracking_server_username
MLFLOW_TRACKING_PASSWORD = tracking_server_password
```
If you do not need users credentials for MLflow leave the fields empty.

### Resources configuration
This package uses a resources configuration to manage the GPU memory limit allowed and the use of cache.
We do not share this file since each machine has its own hardware specifications.
You will need to create a `resources.yaml` file inside the `config` folder with the following structure:

```yaml
GPU:
  MEM_LIMIT: 2048

USE_CACHE: true

```
The GPU memory limit must be specified in Megabytes. If you do not want to set a GPU memory limit, leave the field empty.

__WARNING! The memory limit value is just an example. Do not take it as a reference.__

## Running the experiment
Once the environment is set up, you can run the experiment by executing the following command:

```console
$ python -m run_experiments [-h] [--experiment-name EXPERIMENT_NAME] {local,cloud} {experiment_1.yaml,experiment_2.yaml,experiment_3.yaml}

positional arguments:
  {local,cloud}         The environment to run the profiling in.
  {experiment_1.yaml,experiment_2.yaml,experiment_3.yaml}
                        The name of the configuration file to use.

options:
  -h, --help            show this help message and exit
  --experiment-name EXPERIMENT_NAME
                        The name of the MLflow experiment.
```

The raw measurements for each architecture will be saved in the `data/metrics/raw/{local, cloud}/architecture_name` folder.
If MLflow is enabled, the measurements will also be saved in the MLflow tracking server, together with the trained models.
If not, the trained models will be saved in the `models` folder and the training history will be saved with the raw measurements as `performance-%Y%m%dT%H%M%S.csv`.

You can also train a single model by executing the following command:

```console
$ python -m runner [-h] [--warmup] {local,cloud} {experiment_1.yaml,experiment_2.yaml,experiment_3.yaml} {single-run} ...

positional arguments:
  {local,cloud}         The type of training environment.
  {experiment_1.yaml,experiment_2.yaml,experiment_3.yaml}
                        The configuration file to use.
  {single-run}          sub-command help
    single-run          Single training help

options:
  -h, --help            show this help message and exit
  --warmup              Warmup the GPU.
```

The training history and the model will be saved following the same rules as the profiling script.

### Training data
We do not share the Chesslive dataset. However, you can use the Caltech101 and Stanford Dogs datasets to train the Inception V3 model. To use this datasets with the rest of the models you can extend each of the models' base class located in [src/models/dl](src/models/dl/) and adding it to the [model_factory](src/models/dl/model_factory.py).

## Collected data
All the data collected and produced during the study can be found in the `data.zip` file in the [Relseases](https://github.com/GAISSA-UPC/dl-training-energy-estimation/releases) section.

The folder is expected to be extracted at the root of the project and the metrics collected can be found inside the `data/metrics` folder. The data is organized in the following structure:

```
.
├── - auxiliary
├── - raw
├── - interim
└── - processed
```

The `auxiliary` folder contains a the list of raw measurements that have been processed. These are used to speed up the processing of new raw data.
The `raw` folder contains the raw measurements collected during the experiment.
The `interim` folder contains the processed data that is used to generate the final dataset.
The `processed` folder contains the final data used to perform the analysis.

## Data analysis
The data analysis is done using [Jupyter Notebooks](https://jupyter.org/). You can find the analysis inside the `notebooks` folder. All the plots generated are saved in the `out/figures` folder.

## License
The software under this project is licensed under the terms of the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information.

The data used in this project is licensed under the terms of the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. See the LICENSE in the `data/LICENSE` file for more information.
