from abc import ABC, abstractmethod

from numpy import ndarray

from src.modelling.data_model import Data
from sklearn.base import BaseEstimator

import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def create_model(self) -> BaseEstimator:
        ...

    def train(self, data: Data) -> None:
        if self.model is None:
            self.model = self.create_model()

        if not hasattr(self.model, 'fit') or not callable(getattr(self.model, 'fit')):
            raise TypeError("The created model must have a callable `fit` method.")

        self.model.fit(Data.get_X_train(), Data.get_Y_train())

    def predict(self, data: np.ndarray) -> ndarray:
        if self.model is None:
            self.model = self.create_model()

        if not hasattr(self.model, 'predict') or not callable(getattr(self.model, 'predict')):
            raise TypeError("The created model must have a callable `predict` method.")

        return self.model.predict(data)

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        "todo"
        ...