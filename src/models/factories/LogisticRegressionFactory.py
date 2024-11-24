from src.models.classifiers.LogisticRegression import LogisticRegressionModel
from src.models.factories.ModelFactory import ModelFactory
from src.models.strategies.logisticregression_strategy import LogisticRegressionStrategy


class LogisticRegressionFactory(ModelFactory):
    def create_model(self, model_name: str, load_saved: bool) -> LogisticRegressionStrategy:
        model =  LogisticRegressionModel(model_name, load_saved)
        return LogisticRegressionStrategy(model)