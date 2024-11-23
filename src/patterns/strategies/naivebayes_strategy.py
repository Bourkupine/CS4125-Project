from models.classifiers.NaiveBayes import NaiveBayes
from patterns.model_strategy import ModelStrategyInterface
from src.modelling.data_model import Data

class NaiveBayesStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = NaiveBayes()

    def train(self, data: Data):
        X_train = data.get_X_train()
        y_train = data.get_Y_train()
        self.model.train(X_train, y_train)

    def predict(self, data: Data):
        X_test = data.get_X_test()
        self.model.predict(X_test)
