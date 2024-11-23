from models.classifiers.NaiveBayes import NaiveBayes
from patterns.model_strategy import ModelStrategyInterface

class NaiveBayesStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = NaiveBayes()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        self.model.predict(X_test)
