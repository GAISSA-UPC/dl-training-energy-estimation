"""
Module to define models using transfer learning from InceptionV3 for different datasets.
"""

from abc import ABC

from tensorflow.keras import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten

from src.data.datasets import (
    Caltech101Dataset,
    Cifar10Dataset,
    Food101Dataset,
    StanfordDogsDataset,
)
from src.models.dl.model import BaseModel


class InceptionV3Model(BaseModel, ABC):
    """
    Base model using the InceptionV3 architecture.
    """

    ARCHITECTURE = "inception_v3"

    # Customize at wil
    INITIAL_EPOCHS = 200
    FINE_TUNING_EPOCHS = 0

    def __init__(self, *args, input_shape=None, batch_size=None, **kwargs):
        if input_shape is None:
            _input_shape = list(self.DATASET.IMAGE_DIM)
        else:
            _input_shape = list(input_shape)

        _input_shape[0] = max(_input_shape[0], 75)
        _input_shape[1] = max(_input_shape[1], 75)

        self.INPUT_SHAPE = tuple(_input_shape)

        self.BASE_MODEL = InceptionV3(weights="imagenet", include_top=False, input_shape=self.INPUT_SHAPE)
        self.PREPROCESSING_LAYER = preprocess_input

        if batch_size is not None:
            self.BATCH_SIZE = batch_size

        super().__init__(*args, **kwargs)


class InceptionV3Caltech101(InceptionV3Model):
    """
    Model using the InceptionV3 architecture for the Caltech101 dataset.
    """

    LEARNING_RATE = 1e-3
    OPTIMIZER = "adam"
    FREEZE_AT = 1.0
    FT_LEARNING_RATE = 1e-3
    FT_OPTIMIZER = "adam"
    METRICS = ["accuracy", "AUC"]
    LOSS = "categorical_crossentropy"

    DATASET = Caltech101Dataset()

    def __init__(self, *args, input_shape=None, batch_size=None, **kwargs):
        self.CLASSIFIER = Sequential(
            [
                Flatten(),
                Dropout(0.4),
                Dense(2048, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(1024, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(self.DATASET.NUM_CLASSES, activation="softmax"),
            ]
        )
        super().__init__(*args, input_shape=input_shape, batch_size=batch_size, **kwargs)


class InceptionV3StanfordDogs(InceptionV3Model):
    """
    Model using the InceptionV3 architecture for the Stanford Dogs dataset.
    """

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    OPTIMIZER = "adam"
    FREEZE_AT = 1.0
    FT_LEARNING_RATE = 1e-3
    FT_OPTIMIZER = "adam"
    METRICS = ["accuracy", "AUC"]
    LOSS = "categorical_crossentropy"

    DATASET = StanfordDogsDataset()

    def __init__(self, *args, input_shape=None, batch_size=None, **kwargs):
        self.CLASSIFIER = Sequential(
            [
                Flatten(),
                Dropout(0.4),
                Dense(2048, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(2048, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(self.DATASET.NUM_CLASSES, activation="softmax"),
            ]
        )
        super().__init__(*args, input_shape=input_shape, batch_size=batch_size, **kwargs)


class InceptionV3Cifar10(InceptionV3Model):
    """
    Model using the InceptionV3 architecture for the CIFAR-10 dataset.
    """

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    OPTIMIZER = "adam"
    FREEZE_AT = 1.0
    FT_LEARNING_RATE = 1e-3
    FT_OPTIMIZER = "adam"
    METRICS = ["accuracy", "AUC"]
    LOSS = "categorical_crossentropy"

    DATASET = Cifar10Dataset()

    def __init__(self, *args, input_shape=None, batch_size=None, **kwargs):
        self.CLASSIFIER = Sequential(
            [
                Flatten(),
                Dropout(0.4),
                Dense(2048, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(1024, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(self.DATASET.NUM_CLASSES, activation="softmax"),
            ]
        )
        super().__init__(*args, input_shape=input_shape, batch_size=batch_size, **kwargs)


class InceptionV3Food101(InceptionV3Model):
    """
    Model using the InceptionV3 architecture for the Food101 dataset.
    """

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    OPTIMIZER = "adam"
    FREEZE_AT = 1.0
    FT_LEARNING_RATE = 1e-3
    FT_OPTIMIZER = "adam"
    METRICS = ["accuracy", "AUC"]
    LOSS = "categorical_crossentropy"

    DATASET = Food101Dataset()

    def __init__(self, *args, input_shape=None, batch_size=None, **kwargs):
        self.CLASSIFIER = Sequential(
            [
                Flatten(),
                Dropout(0.4),
                Dense(2048, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(1024, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(self.DATASET.NUM_CLASSES, activation="softmax"),
            ]
        )
        super().__init__(*args, input_shape=input_shape, batch_size=batch_size, **kwargs)
