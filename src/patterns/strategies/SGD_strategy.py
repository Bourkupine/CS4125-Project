from models.classifiers.SGD import SGD
from patterns.model_strategy import ModelStrategyInterface

class SGDStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = SGD()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        self.model.predict(X_test)
