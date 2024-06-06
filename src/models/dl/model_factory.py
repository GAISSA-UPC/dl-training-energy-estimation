"""

"""

from abc import ABC, abstractmethod
from enum import Enum

from src.models.dl.inception_resnet_v2 import (
    InceptionResNetV2Caltech101,
    InceptionResNetV2Cifar10,
    InceptionResNetV2Food101,
    InceptionResNetV2Model,
    InceptionResNetV2StanfordDogs,
)
from src.models.dl.inception_v3 import (
    InceptionV3Caltech101,
    InceptionV3Cifar10,
    InceptionV3Food101,
    InceptionV3Model,
    InceptionV3StanfordDogs,
)
from src.models.dl.mobilenet_v2 import MobileNetV2ChessLive, MobileNetV2Model
from src.models.dl.model import BaseModel
from src.models.dl.nasnet_mobile import NASNetMobileChessLive, NASNetMobileModel
from src.models.dl.resnet_50 import ResNet50ChessLive, ResNet50Model
from src.models.dl.vgg16 import VGG16ChessLive, VGG16Model
from src.models.dl.xception import XceptionChessLive, XceptionModel


class Architectures(Enum):
    mobilenet_v2 = "mobilenet_v2"
    nasnet_mobile = "nasnet_mobile"
    xception = "xception"
    resnet50 = "resnet50"
    vgg16 = "vgg16"
    inception_v3 = "inception_v3"
    inception_resnet_v2 = "inception_resnet_v2"

    @staticmethod
    def to_list():
        return [arch.value for arch in Architectures]


class AbstractModelFactory(ABC):
    """
    Abstract model factory.
    """

    @abstractmethod
    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "BaseModel":
        """
        Returns the model for the given dataset.
        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        input_shape : tuple of int
            The input size of the images.
        batch_size : int
            The batch size to use for training.

        Returns
        -------
        model : BaseModel
            The model for the given dataset.
        """
        raise NotImplementedError


class InceptionV3Factory(AbstractModelFactory):
    """
    InceptionV3 model factory.
    """

    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "InceptionV3Model":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        input_shape : tuple
            The input shape of the images.
        batch_size : int
            The batch size to use for training.

        Returns
        -------
        model : InceptionV3Model
            The model for the given dataset.
        """
        if dataset_name == "caltech101":
            return self._get_caltech101_model(input_shape, batch_size)
        elif dataset_name == "stanford_dogs":
            return self._get_stanford_dogs_model(input_shape, batch_size)
        elif dataset_name == "cifar10":
            return self._get_cifar10_model(input_shape, batch_size)
        elif dataset_name == "food101":
            return self._get_food101_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_caltech101_model(input_shape, batch_size):
        return InceptionV3Caltech101(input_shape=input_shape, batch_size=batch_size)

    @staticmethod
    def _get_stanford_dogs_model(input_shape, batch_size):
        return InceptionV3StanfordDogs(input_shape=input_shape, batch_size=batch_size)

    @staticmethod
    def _get_cifar10_model(input_shape, batch_size):
        return InceptionV3Cifar10(input_shape=input_shape, batch_size=batch_size)

    @staticmethod
    def _get_food101_model(input_shape, batch_size):
        return InceptionV3Food101(input_shape=input_shape, batch_size=batch_size)


class InceptionResNetV2Factory(AbstractModelFactory):
    """
    InceptionResNetV2 model factory.
    """

    def get_model(
        self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int
    ) -> "InceptionResNetV2Model":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.
        input_shape: tuple
            The input shape of the images.
        batch_size: int
            The batch size to use for training.

        Returns
        -------
        model: InceptionResNetV2Model
            The model for the given dataset.
        """
        if dataset_name == "caltech101":
            return self._get_caltech101_model(input_shape, batch_size)
        elif dataset_name == "stanford_dogs":
            return self._get_stanford_dogs_model(input_shape, batch_size)
        elif dataset_name == "cifar10":
            return self._get_cifar10_model(input_shape, batch_size)
        elif dataset_name == "food101":
            return self._get_food101_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_caltech101_model(input_shape, batch_size):
        return InceptionResNetV2Caltech101(input_shape=input_shape, batch_size=batch_size)

    @staticmethod
    def _get_stanford_dogs_model(input_shape, batch_size):
        return InceptionResNetV2StanfordDogs(input_shape=input_shape, batch_size=batch_size)

    @staticmethod
    def _get_cifar10_model(input_shape, batch_size):
        return InceptionResNetV2Cifar10(input_shape=input_shape, batch_size=batch_size)

    @staticmethod
    def _get_food101_model(input_shape: tuple, batch_size: int):
        return InceptionResNetV2Food101(input_shape=input_shape, batch_size=batch_size)


class VGG16Factory(AbstractModelFactory):
    """
    VGG16 model factory.
    """

    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "VGG16Model":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.
        input_shape: tuple of int or None
            The input shape of the images.
        batch_size: int or None
            The batch size to use for training.

        Returns
        -------
        model: VGG16Model
            The model for the given dataset.
        """
        if dataset_name == "chesslive":
            return self._get_chesslive_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_chesslive_model(input_shape, batch_size):
        return VGG16ChessLive(input_shape=input_shape, batch_size=batch_size)


class XceptionFactory(AbstractModelFactory):
    """
    Xception model factory.
    """

    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "XceptionModel":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.
        input_shape: tuple of int or None
            The input shape of the images.
        batch_size: int or None
            The batch size to use for training.

        Returns
        -------
        model: XceptionModel
            The model for the given dataset.
        """
        if dataset_name == "chesslive":
            return self._get_chesslive_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_chesslive_model(input_shape, batch_size):
        return XceptionChessLive(input_shape=input_shape, batch_size=batch_size)


class ResNet50Factory(AbstractModelFactory):
    """
    ResNet50 model factory.
    """

    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "ResNet50Model":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.
        input_shape: tuple of int or None
            The input shape of the images.
        batch_size: int or None
            The batch size to use for training.

        Returns
        -------
        model: ResNet50Model
            The model for the given dataset.
        """
        if dataset_name == "chesslive":
            return self._get_chesslive_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_chesslive_model(input_shape, batch_size):
        return ResNet50ChessLive(input_shape=input_shape, batch_size=batch_size)


class MobileNetV2Factory(AbstractModelFactory):
    """
    MobileNetV2 model factory.
    """

    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "MobileNetV2Model":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.
        input_shape: tuple of int or None
            The input shape of the images.
        batch_size: int or None
            The batch size to use for training.

        Returns
        -------
        model: ResNet50Model
            The model for the given dataset.
        """
        if dataset_name == "chesslive":
            return self._get_chesslive_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_chesslive_model(input_shape, batch_size):
        return MobileNetV2ChessLive(input_shape=input_shape, batch_size=batch_size)


class NASNetMobileFactory(AbstractModelFactory):
    """
    NASNet Mobile model factory.
    """

    def get_model(self, dataset_name: str, input_shape: tuple[int, int, int], batch_size: int) -> "NASNetMobileModel":
        """
        Returns the model for the given dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset.
        input_shape: tuple of int or None
            The input shape of the images.
        batch_size: int or None
            The batch size to use for training.

        Returns
        -------
        model: ResNet50Model
            The model for the given dataset.
        """
        if dataset_name == "chesslive":
            return self._get_chesslive_model(input_shape, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def _get_chesslive_model(input_shape, batch_size):
        return NASNetMobileChessLive(input_shape=input_shape, batch_size=batch_size)
