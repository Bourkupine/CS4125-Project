from patterns.model_strategy import ModelStrategyInterface

class ModelContext:
    def __init__(self, strategy: ModelStrategyInterface):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelStrategyInterface):
        self.strategy = strategy

    def train(self, X_train, y_train):
        self.strategy.train(X_train, y_train)

    def predict(self, X_test):
        self.strategy.predict(X_test)
