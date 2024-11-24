import numpy as np

from src.models.classifiers.BaseModel import BaseModel
from src.models.strategies.model_strategy import ModelStrategy
from src.modelling.data_model import Data

class NaiveBayesStrategy(ModelStrategy):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def train(self, data: Data, save: bool):
        self.model.train(data, save)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
