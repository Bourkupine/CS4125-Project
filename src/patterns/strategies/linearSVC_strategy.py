from models.classifiers.LinearSVC import LinearSVC
from patterns.model_strategy import ModelStrategyInterface

class LinearSVCStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = LinearSVC()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        self.model.predict(X_test)
