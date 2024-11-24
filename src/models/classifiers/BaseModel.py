from abc import ABC, abstractmethod

import numpy as np
from joblib import dump, load
from numpy import ndarray
from sklearn.base import BaseEstimator

from src.modelling.data_model import Data


class BaseModel(ABC):
    def __init__(self, name: str, load_model: bool):
        self.name = name
        self.path = f"./trained_models/{self.name}"
        self.load: bool = load_model
        self.model: BaseEstimator | None = None

    @abstractmethod
    def create_model(self):
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
            print(f"No existing Model found for {self.name} at {self.path}, creating new model. Make sure to train it before evaluation.")
            return None
