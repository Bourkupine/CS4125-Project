from models.classifiers.RandomForest import RandomForest
from patterns.model_strategy import ModelStrategyInterface

class RandomForestStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = RandomForest()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        self.model.predict(X_test)
