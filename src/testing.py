import datetime
import os

import pandas as pd
from mlflow import MlflowClient
from tqdm import tqdm

from features.preprocessing import FLOPS_TO_GFLOPS
from models.dl.model_factory import (
    InceptionV3Factory,
    MobileNetV2Factory,
    NASNetMobileFactory,
    ResNet50Factory,
    VGG16Factory,
    XceptionFactory,
)
from profiling.metrics import compute_maccs
from src.environment import METRICS_DIR


def add_total_memory():
    input_dir = METRICS_DIR / "raw"
    for train_environment in os.listdir(input_dir):
        architectures_folder = input_dir / train_environment
        for architecture in os.listdir(architectures_folder):
            datasets_folder = architectures_folder / architecture
            for dataset in os.listdir(datasets_folder):
                if (datasets_folder / dataset).is_file():
                    continue
                cpu_files = (datasets_folder / dataset).glob(r"cpu*.csv")
                cpu_files = sorted(cpu_files, key=lambda x: os.path.basename(x).split("-")[-1])
                if train_environment in ["local", "local-v2"]:
                    total_memory = [32] * len(cpu_files)
                else:
                    if architecture == "inception_v3" and dataset == "stanford_dogs":
                        total_memory = [16] * len(cpu_files[:-120]) + [8] * 30 + [24] * 30 + [4] * 30 + [32] * 30
                    else:
                        total_memory = [16] * len(cpu_files)
                for file, memory in zip(cpu_files, total_memory):
                    try:
                        df = pd.read_csv(file)
                        if "total memory (GB)" in df.columns:
                            continue
                        if "total memory (MB)" in df.columns:
                            df.drop(columns=["total memory (MB)"], inplace=True)
                        df["total memory (GB)"] = memory
                        df.to_csv(file, index=False)
                    except pd.errors.EmptyDataError:
                        continue


def build_epoch_ends_from_mlflow():
    aggregated_metrics = pd.read_parquet(
        METRICS_DIR / "processed" / "dl-training-energy-consumption-dataset.gzip"
    ).sort_values(by=["start time"])
    # cloud_inception_runs = aggregated_metrics.query(
    raw_metrics = pd.read_parquet(
        METRICS_DIR / "interim" / "dl-training-profiling-dataset.gzip", columns=["run_id", "creation_time"]
    ).drop_duplicates()
    experiment_data = aggregated_metrics.loc[
        :, ["training environment", "architecture", "dataset", "run_id", "mlflow run id", "return code"]
    ].drop_duplicates()
    experiment_data = experiment_data.merge(raw_metrics, on="run_id", how="left")

    client = MlflowClient()
    training_environments = {"Cloud": "cloud", "Local Normal User": "local", "Local ML Engineer/Gamer": "local-v2"}
    files_with_epoch_ends = {"training environment": [], "architecture": [], "dataset": []}
    new_epoch_ends = {"training environment": [], "architecture": [], "dataset": []}
    runs_without_epoch_ends = {"training environment": [], "architecture": [], "dataset": [], "return code": []}
    for _, row in tqdm(experiment_data.iterrows(), total=experiment_data.shape[0]):
        training_environment = training_environments[row["training environment"]]
        architecture = row["architecture"]
        if architecture == "inception_v3" and training_environment == "local-v2":
            training_environment = "local"
        dataset = row["dataset"]
        creation_time = row["creation_time"]

        epoch_ends_file = (
            METRICS_DIR / "raw" / training_environment / architecture / dataset / f"epoch_end-{creation_time}.csv"
        )
        if os.path.exists(epoch_ends_file):
            files_with_epoch_ends["training environment"].append(row["training environment"])
            files_with_epoch_ends["architecture"].append(row["architecture"])
            files_with_epoch_ends["dataset"].append(row["dataset"])
            continue

        mlflow_id = row["mlflow run id"]

        history = client.get_metric_history(mlflow_id, "val_loss")
        if not history:
            runs_without_epoch_ends["training environment"].append(row["training environment"])
            runs_without_epoch_ends["architecture"].append(row["architecture"])
            runs_without_epoch_ends["dataset"].append(row["dataset"])
            runs_without_epoch_ends["return code"].append(row["return code"])
            continue

        epoch_ends = [(x.step, x.timestamp) for x in history]
        epoch_ends = (
            pd.DataFrame(epoch_ends, columns=["epoch", "end_time"]).sort_values(by=["epoch"]).reset_index(drop=True)
        )
        epoch_ends["end_time"] = pd.to_datetime(epoch_ends["end_time"], unit="ms", origin="unix")
        # Add one hour to account for timezone difference
        epoch_ends["end_time"] = epoch_ends["end_time"] + datetime.timedelta(hours=1)

        # Save all but last epoch end. Last epoch is not saved because it corresponds to the best epoch due to EarlyStopping.
        epoch_ends.iloc[:-1].to_csv(epoch_ends_file, index=False)
        new_epoch_ends["training environment"].append(row["training environment"])
        new_epoch_ends["architecture"].append(row["architecture"])
        new_epoch_ends["dataset"].append(row["dataset"])

    existing_files_df = pd.DataFrame(files_with_epoch_ends)
    new_files_df = pd.DataFrame(new_epoch_ends)

    print("Existing files:")
    print(existing_files_df.groupby(by=["training environment", "architecture", "dataset"]).size())
    print("")
    print("New files:")
    print(new_files_df.groupby(by=["training environment", "architecture", "dataset"]).size())
    print("")
    print("Runs without epoch ends:")
    print(
        pd.DataFrame(runs_without_epoch_ends)
        .groupby(by=["training environment", "architecture", "dataset", "return code"])
        .size()
    )
    print("")
    print("Total files:")
    print(experiment_data.groupby(by=["training environment", "architecture", "dataset"]).size())


if __name__ == "__main__":
    # add_total_memory()
    # build_metrics_dataset(save_to_file=True)
    # build_analysis_dataset(save_to_file=True)
    # build_epoch_energy_dataset(save_to_file=True)
    # aggregated_metrics = pd.read_parquet(
    #     METRICS_DIR / "processed" / "clean-dl-training-energy-consumption-dataset.gzip"
    # ).sort_values(by=["start time"])
    # metrics = pd.read_parquet(os.path.join(METRICS_DIR, "interim", "dl-training-profiling-dataset.gzip"))
    # m = 10
    # L = m
    # regimes_df, profiles = find_stabilizing_point(aggregated_metrics["run_id"].unique(), metrics, m, L, save=True)
    # cloud_inception_runs = aggregated_metrics.query(
    #     "`training environment` == 'Cloud' and architecture == 'inception_v3'"
    # )["run_id"].unique()
    # metrics = metrics.query("run_id in @cloud_inception_runs")
    # metrics["elapsed_time"] = metrics["elapsed_time"] / np.timedelta64(1, "s")

    # train_timeseries_kmeans(metrics, n_clusters=10, metric="dtw", max_iter=5, n_init=2, verbose=True, n_jobs=12)
    # build_epoch_ends_from_mlflow()
    # _process_raw_files(
    #     "cloud",
    #     "mobilenet_v2",
    #     "chesslive-occupancy",
    #     "00880f78cd0f458ba2a0d2ea72c88805",
    #     "/home/santiago/Local-Projects/seaa2023_ect_extension/data/metrics/raw/cloud/mobilenet_v2/chesslive-occupancy/cpu-mem-usage-20221206T003424.csv",
    #     "/home/santiago/Local-Projects/seaa2023_ect_extension/data/metrics/raw/cloud/mobilenet_v2/chesslive-occupancy/gpu-power-20221206T003424.csv",
    # )
    model = MobileNetV2Factory().get_model("chesslive", (128, 128, 3), 32).build_model()
    # print(model.model.summary())
    mobilenet_flops = compute_maccs(model.model) * 2 * FLOPS_TO_GFLOPS
    model = NASNetMobileFactory().get_model("chesslive", (128, 128, 3), 32).build_model()
    # print(model.model.summary())
    nasnet_flops = compute_maccs(model.model) * 2 * FLOPS_TO_GFLOPS
    model = XceptionFactory().get_model("chesslive", (128, 128, 3), 32).build_model()
    # print(model.model.summary())
    xception_flops = compute_maccs(model.model) * 2 * FLOPS_TO_GFLOPS
    model = ResNet50Factory().get_model("chesslive", (128, 128, 3), 32).build_model()
    # print(model.model.summary())
    resnet_flos = compute_maccs(model.model) * 2 * FLOPS_TO_GFLOPS
    model = VGG16Factory().get_model("chesslive", (128, 128, 3), 32).build_model()
    # print(model.model.summary())
    vgg16_flops = compute_maccs(model.model) * 2 * FLOPS_TO_GFLOPS
    model = InceptionV3Factory().get_model("caltech101", (128, 128, 3), 32).build_model()
    # print(model.model.summary())
    inception_flops = compute_maccs(model.model) * 2 * FLOPS_TO_GFLOPS

    print(f"MobileNetV2: {mobilenet_flops:.2f} GFLOPS")
    print(f"NASNetMobile: {nasnet_flops:.2f} GFLOPS")
    print(f"Xception: {xception_flops:.2f} GFLOPS")
    print(f"ResNet50: {resnet_flos:.2f} GFLOPS")
    print(f"VGG16: {vgg16_flops:.2f} GFLOPS")
    print(f"InceptionV3: {inception_flops:.2f} GFLOPS")
