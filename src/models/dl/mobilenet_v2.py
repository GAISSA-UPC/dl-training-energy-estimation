"""
Module to define models using transfer learning from MobileNet V2 for different datasets.
"""

from abc import ABC

from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from src.data.datasets import ChessliveDataset
from src.models.dl.model import BaseModel


class MobileNetV2Model(BaseModel, ABC):
    """
    Base model using the MobileNetV2 architecture.
    """

    ARCHITECTURE = "mobilenet_v2"

    # Customize at wil
    INITIAL_EPOCHS = 10
    FINE_TUNING_EPOCHS = 190

    def __init__(self, *args, weights="imagenet", input_shape=None, batch_size=None, **kwargs):
        if input_shape is None:
            _input_shape = list(self.DATASET.IMAGE_DIM)
        else:
            _input_shape = list(input_shape)

        _input_shape[0] = max(_input_shape[0], 75)
        _input_shape[1] = max(_input_shape[1], 75)

        self.INPUT_SHAPE = tuple(_input_shape)

        self.BASE_MODEL = MobileNetV2(weights=weights, include_top=False, input_shape=self.INPUT_SHAPE)
        self.PREPROCESSING_LAYER = preprocess_input

        if batch_size is not None:
            self.BATCH_SIZE = batch_size

        super().__init__(*args, **kwargs)


class MobileNetV2ChessLive(MobileNetV2Model):
    """
    Model using the MobileNetV2 architecture for the ChessLive dataset.
    """

    LEARNING_RATE = 1e-4
    OPTIMIZER = "adam"
    FREEZE_AT = 0.0
    FT_LEARNING_RATE = 1e-5
    FT_OPTIMIZER = "sgd"
    FT_MOMENTUM = 0.9
    METRICS = ["accuracy", "Precision", "Recall", "AUC"]
    LOSS = "binary_crossentropy"

    DATASET = ChessliveDataset()

    def __init__(self, *args, weights="imagenet", input_shape=None, batch_size=None, **kwargs):
        self.CLASSIFIER = Sequential(
            [
                GlobalAveragePooling2D(),
                Dense(512, activation="relu"),
                Dense(1, activation="softmax"),
            ]
        )
        super().__init__(*args, weights=weights, input_shape=input_shape, batch_size=batch_size, **kwargs)
