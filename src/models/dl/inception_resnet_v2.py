"""
Module to define models using transfer learning from Inception-ResNet V2 for different datasets.
"""

from abc import ABC

from tensorflow.keras import Sequential
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten

from src.data.datasets import (
    Caltech101Dataset,
    Cifar10Dataset,
    Food101Dataset,
    StanfordDogsDataset,
)
from src.models.dl.model import BaseModel


class InceptionResNetV2Model(BaseModel, ABC):
    """
    Base model using the InceptionResNetV2 architecture.
    """

    ARCHITECTURE = "inception_resnet_v2"

    # Customize at will
    INITIAL_EPOCHS = 200
    FINE_TUNING_EPOCHS = 0

    def __init__(self, input_shape: tuple = None, batch_size: int = None):
        """
        Parameters
        ----------
        input_shape: tuple
            The input shape of the images.
        batch_size: int
            The batch size to use for training.
        """
        if input_shape is None:
            _input_shape = list(self.DATASET.IMAGE_DIM)
        else:
            _input_shape = list(input_shape)

        _input_shape[0] = max(_input_shape[0], 75)
        _input_shape[1] = max(_input_shape[1], 75)

        self.INPUT_SHAPE = tuple(_input_shape)

        self.BASE_MODEL = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=self.INPUT_SHAPE)
        self.PREPROCESSING_LAYER = preprocess_input

        if batch_size is not None:
            self.BATCH_SIZE = batch_size
        super().__init__()


class InceptionResNetV2Caltech101(InceptionResNetV2Model):
    """
    Model using the InceptionResNetV2 architecture for the Caltech101 dataset.
    """

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    OPTIMIZER = "adam"
    FREEZE_AT = 1.0
    FT_LEARNING_RATE = 1e-3
    FT_OPTIMIZER = "adam"
    METRICS = ["accuracy", "AUC"]
    LOSS = "categorical_crossentropy"

    DATASET = Caltech101Dataset()
    CLASSIFIER = Sequential(
        [
            Flatten(),
            Dropout(0.4),
            Dense(2048, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1024, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(DATASET.NUM_CLASSES, activation="softmax"),
        ]
    )

    def __init__(self, input_shape: tuple = None, batch_size: int = None):
        super().__init__(input_shape=input_shape, batch_size=batch_size)


class InceptionResNetV2StanfordDogs(InceptionResNetV2Model):
    """
    Model using the InceptionResNetV2 architecture for the Stanford Dogs dataset.
    """

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    OPTIMIZER = "adam"
    FREEZE_AT = 1.0
    FT_LEARNING_RATE = 1e-3
    FT_OPTIMIZER = "adam"
    METRICS = ["accuracy", "AUC"]
    LOSS = "categorical_crossentropy"

    DATASET = StanfordDogsDataset()
    CLASSIFIER = Sequential(
        [
            Flatten(),
            Dropout(0.4),
            Dense(2048, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1024, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(DATASET.NUM_CLASSES, activation="softmax"),
        ]
    )

    def __init__(self, input_shape: tuple = None, batch_size: int = None):
        super().__init__(input_shape=input_shape, batch_size=batch_size)


class InceptionResNetV2Cifar10(InceptionResNetV2Model):
    """
    Model using the InceptionResNetV2 architecture for the CIFAR-10 dataset.
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
    CLASSIFIER = Sequential(
        [
            Flatten(),
            Dropout(0.4),
            Dense(2048, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1024, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(DATASET.NUM_CLASSES, activation="softmax"),
        ]
    )

    def __init__(self, input_shape: tuple = None, batch_size: int = None):
        super().__init__(input_shape=input_shape, batch_size=batch_size)


class InceptionResNetV2Food101(InceptionResNetV2Model):
    """
    Model using the InceptionResNetV2 architecture for the Food101 dataset.
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
    CLASSIFIER = Sequential(
        [
            Flatten(),
            Dropout(0.4),
            Dense(2048, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1024, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(DATASET.NUM_CLASSES, activation="softmax"),
        ]
    )

    def __init__(self, input_shape: tuple = None, batch_size: int = None):
        super().__init__(input_shape=input_shape, batch_size=batch_size)
