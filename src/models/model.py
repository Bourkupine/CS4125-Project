from abc import ABC, abstractmethod

from sklearn.ensemble import BaseEnsemble

import src.utils as utils
from src.modelling.data_model import Data
from sklearn.base import BaseEstimator

import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self, model: BaseEnsemble) -> None:
        self.model = model

    def create_model(self) -> BaseEstimator:
        ...

    @abstractmethod
    def train(self, data: Data) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """

        if self.model is None:
            self.model = self.create_model()

        if not hasattr(self.model, 'fit') or not callable(getattr(self.model, 'fit')):
            raise TypeError("The created model must have a callable `fit` method.")

        self.model.fit()

    @abstractmethod
    def predict(self) -> int:
        """

        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
