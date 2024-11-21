from abc import ABC, abstractmethod

'''
Strategy Pattern:
'''

class StrategyInterface(ABC):

    '''
    This is an abstract method for training our model
    '''
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    '''
    This is an abstract method for predicting using our model
    '''
    @abstractmethod
    def predict(self, X_test):
        pass
