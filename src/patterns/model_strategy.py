from abc import ABC, abstractmethod
from src.modelling.data_model import Data

'''
Strategy Pattern:
'''

class ModelStrategyInterface(ABC):

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
    def predict(self, data: Data):
        pass
