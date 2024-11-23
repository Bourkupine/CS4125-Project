from models.classifiers.AdaBoost import AdaBoostClassifier
from patterns.model_strategy import ModelStrategyInterface

class AdaBoostStrategy(ModelStrategyInterface):
    def __init__(self):
        self.model = AdaBoostClassifier()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        self.model.predict(X_test)
