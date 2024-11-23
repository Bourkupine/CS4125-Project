from abc import ABC, abstractmethod

from numpy import ndarray
from joblib import dump, load

from src.modelling.data_model import Data
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.path = f"./trained_models/{self.name}"
        model = load(self.path)
        self.model = model if model else self.create_model()

    @abstractmethod
    def create_model(self) -> BaseEstimator:
        ...

    def train(self, data: Data, save: bool = False) -> None:
        if not hasattr(self.model, 'fit') or not callable(getattr(self.model, 'fit')):
            raise TypeError("The created model must have a callable `fit` method.")

        self.model.fit(data.get_X_train(), data.get_Y_train())
        if save:
            self.save_model()

    def predict(self, data: np.ndarray) -> ndarray:
        if not hasattr(self.model, 'predict') or not callable(getattr(self.model, 'predict')):
            raise TypeError("The created model must have a callable `predict` method.")

        return self.model.predict(data)

    def save_model(self):
        try:
            dump(self.model, self.path)
        except Exception as e:
            # Move to logger in future
            print(f"Couldn't save model {self.name} to file {self.path}. Error: {e}")

    def load_model(self) -> BaseEstimator | None:
        try:
            return load(self.path)
        except FileNotFoundError:
            # Move to logger in future
            print(
                f"No existing Model found for {self.name} at {self.path}, creating new model. Make sure to train it before evaluation.")
            return None
