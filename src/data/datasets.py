"""
Configuration for datasets.
"""

from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import image_dataset_from_directory

from src.environment import DATASET_DIR, SEED


class Datasets(Enum):
    CALTECH101 = "caltech101"
    STANFORD_DOGS = "stanford_dogs"
    CIFAR10 = "cifar10"
    FOOD101 = "food101"
    CHESSLIVE = "chesslive"

    @staticmethod
    def to_list():
        return [dataset.value for dataset in Datasets]


class BaseDatasetConfig(ABC):
    """
    Base configuration for datasets.
    """

    NAME: str = None
    NUM_CLASSES: int = 0
    TRAIN_SIZE: float = 0.0
    VAL_SIZE: float = 0.0

    IMAGE_DIM: tuple = ()
    CENTER_CROP: bool = False

    USE_CACHE: bool = False

    def load_dataset(self, image_dim: tuple, batch_size: int):
        self.IMAGE_DIM = image_dim
        train_ds, val_ds = self._load_data()

        self.TRAIN_SIZE = train_ds.cardinality().numpy()
        self.VAL_SIZE = val_ds.cardinality().numpy()
        train_ds = self.prepare_data(train_ds, batch_size, augment=True)
        val_ds = self.prepare_data(val_ds, batch_size)
        return train_ds, val_ds

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def prepare_data(self, dataset, batch_size, augment=False):
        # Resize and rescale all datasets.
        dataset = dataset.map(lambda x, y: (self.resize_and_rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch all datasets.
        dataset = dataset.batch(batch_size)

        if self.USE_CACHE:
            dataset = dataset.cache()

        # Use data augmentation only on the training set.
        if augment:
            dataset = dataset.repeat().map(
                lambda x, y: (self.data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
            )

        # Use buffered prefetching on all datasets.
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def resize_and_rescale(self, image):
        # Do not resclase as it interferes with the base model preprocessing
        # image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, self.IMAGE_DIM[:-1])

    def data_augmentation(self, image):
        image = tf.image.random_flip_left_right(image)
        if self.CENTER_CROP:
            image = tf.image.random_crop(image, size=self.IMAGE_DIM)
            image = tf.image.resize(image, self.IMAGE_DIM[:-1])
        return image


class ChessliveDataset(BaseDatasetConfig):
    NAME = "chesslive"
    NUM_CLASSES = 2
    IMAGE_DIM = (128, 128, 3)
    DATA_FOLDER = DATASET_DIR
    TEST_SPLIT = 0.3

    def _load_data(self):
        train_ds = image_dataset_from_directory(
            self.DATA_FOLDER,
            validation_split=self.TEST_SPLIT,
            subset="training",
            shuffle=True,
            seed=SEED,
            image_size=self.IMAGE_DIM[:-1],
            batch_size=None,
            label_mode="binary",
        )
        val_ds = image_dataset_from_directory(
            self.DATA_FOLDER,
            validation_split=self.TEST_SPLIT,
            subset="validation",
            shuffle=True,
            seed=SEED,
            image_size=self.IMAGE_DIM[:-1],
            batch_size=None,
            label_mode="binary",
        )
        num_classes = len(train_ds.class_names)
        return train_ds, val_ds, num_classes

    def data_augmentation(self, image):
        image = super().data_augmentation(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.5, 2.0)
        image = tf.image.random_saturation(image, 0.75, 1.25)
        image = tf.image.random_hue(image, 0.1)

        return tf.clip_by_value(image, 0.0, 1.0)  # Clip to avoid values out of range after random_brightness


class TensorflowDatasetConfig(BaseDatasetConfig):
    """
    Configuration for TensorFlow datasets.
    """

    SPLIT_NAMES: list = []

    def _load_data(self):
        (train, test), _ = tfds.load(
            self.NAME,
            split=self.SPLIT_NAMES,
            as_supervised=True,
            shuffle_files=True,
            with_info=True,
            read_config=tfds.ReadConfig(shuffle_seed=SEED),
        )

        # Convert labels to one-hot encoding
        train_ds = train.map(lambda x, y: (x, tf.one_hot(y, self.NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test.map(lambda x, y: (x, tf.one_hot(y, self.NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)

        return train_ds, test_ds

    # def data_augmentation(self, image):
    # image = super().data_augmentation(image)
    # image = tf.image.random_brightness(image, 0.2)

    # return tf.clip_by_value(image, 0.0, 1.0)


class Caltech101Dataset(TensorflowDatasetConfig):
    """
    Configuration for Caltech101 dataset.
    """

    NAME = "caltech101"
    NUM_CLASSES = 102
    IMAGE_DIM = (224, 224, 3)
    USE_CENTER_CROP = True
    SPLIT_NAMES = ["train", "test"]
    USE_CACHE = True


class StanfordDogsDataset(TensorflowDatasetConfig):
    """
    Configuration for Stanford Dogs dataset.
    """

    NAME = "stanford_dogs"
    NUM_CLASSES = 120
    IMAGE_DIM = (200, 200, 3)
    USE_CENTER_CROP = True
    SPLIT_NAMES = ["train", "test"]
    USE_CACHE = True


class Cifar10Dataset(TensorflowDatasetConfig):
    """
    Configuration for CIFAR-10 dataset.
    """

    NAME = "cifar10"
    NUM_CLASSES = 10
    IMAGE_DIM = (32, 32, 3)
    USE_CENTER_CROP = True
    SPLIT_NAMES = ["train", "test"]
    USE_CACHE = True


class Food101Dataset(TensorflowDatasetConfig):
    """
    Configuration for Food101 dataset.
    """

    NAME = "food101"
    NUM_CLASSES = 101
    IMAGE_DIM = (224, 224, 3)
    USE_CENTER_CROP = True
    SPLIT_NAMES = ["train", "validation"]
    USE_CACHE = False
