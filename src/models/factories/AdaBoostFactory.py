from src.models.classifiers.AdaBoost import AdaBoostModel
from src.models.factories.ModelFactory import ModelFactory
from src.models.strategies.adaboost_strategy import AdaBoostStrategy


class AdaBoostFactory(ModelFactory):
    def create_model(self, model_name: str, load_saved: bool) -> AdaBoostStrategy:
        model =  AdaBoostModel(model_name, load_saved)
        return AdaBoostStrategy(model)
