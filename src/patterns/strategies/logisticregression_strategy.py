from models.classifiers.LogisticRegression import LogisticRegression
from patterns.model_strategy import ModelStrategyInterface

class LogisticRegressionStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        self.model.predict(X_test)
