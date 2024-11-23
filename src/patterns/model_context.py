from patterns.model_strategy import ModelStrategyInterface
from src.modelling.data_model import Data

class ModelContext:
    def __init__(self, strategy: ModelStrategyInterface):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelStrategyInterface):
        self.strategy = strategy

    def train(self, data: Data):
        X_train = data.get_X_train()
        y_train = data.get_Y_train()
        self.strategy.train(X_train, y_train)

    def predict(self, data: Data):
        X_test = data.get_X_test()
        self.strategy.predict(X_test)
