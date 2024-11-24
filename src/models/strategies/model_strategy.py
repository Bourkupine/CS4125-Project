from abc import ABC, abstractmethod

import numpy as np

from src.modelling.data_model import Data
from src.models.classifiers.BaseModel import BaseModel

'''
Strategy Pattern:
'''

class ModelStrategy(ABC):
    def __init__(self, model: BaseModel):
        self.model = model
    '''
    This is an abstract method for training our model
    '''
    @abstractmethod
    def train(self, data: Data):
        pass

    '''
    This is an abstract method for predicting using our model
    '''
    @abstractmethod
    def predict(self, data: Data) -> np.ndarray:
        pass
