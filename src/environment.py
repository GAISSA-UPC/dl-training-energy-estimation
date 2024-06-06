import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils import read_config

ROOT = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "config"

# The configuration file to use for the experiments
CONFIG_FILE = CONFIGS_DIR / os.getenv("CONFIG_FILE", "experiment_1.yaml")

load_dotenv()

resources_cfg = read_config(CONFIGS_DIR / "resources.yaml")
GPU_MEM_LIMIT = resources_cfg["GPU"]["MEM_LIMIT"]
USE_CACHE = resources_cfg["USE_CACHE"]

base_cfg = read_config(CONFIGS_DIR / "base.yaml")
OUTPUT_DIR = ROOT / base_cfg["OUTPUT_DIR"]
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORTS_DIR = ROOT / base_cfg["REPORTS_DIR"]
os.makedirs(REPORTS_DIR, exist_ok=True)
FIGURES_DIR = ROOT / base_cfg["FIGURES_DIR"]
os.makedirs(FIGURES_DIR, exist_ok=True)
DATA_DIR = ROOT / base_cfg["DATA_DIR"]
os.makedirs(DATA_DIR, exist_ok=True)
METRICS_DIR = ROOT / base_cfg["METRICS_DIR"]
os.makedirs(METRICS_DIR, exist_ok=True)
DATASET_DIR = ROOT / base_cfg["DATASET_DIR"]
os.makedirs(DATASET_DIR, exist_ok=True)
MODELS_DIR = ROOT / base_cfg["MODELS_DIR"]
os.makedirs(MODELS_DIR, exist_ok=True)
PRETRAINED_MODELS_DIR = MODELS_DIR / "pre-trained"
os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)

INFERENCE_MODEL = MODELS_DIR / base_cfg["INFERENCE_MODEL"]

DEBUG = base_cfg["DEBUG"]

SEED = 2022


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)
    logger.addHandler(handler)
    return logger


logger = setup_custom_logger("profiler")
