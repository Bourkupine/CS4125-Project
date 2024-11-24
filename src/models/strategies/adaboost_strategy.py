import numpy as np

from src.modelling.data_model import Data
from src.models.classifiers.BaseModel import BaseModel
from src.models.strategies.model_strategy import ModelStrategy


class AdaBoostStrategy(ModelStrategy):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def train(self, data: Data):
        self.model.train(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
