from src.models.classifiers.NaiveBayesModel import NaiveBayesModel
from src.models.factories.ModelFactory import ModelFactory
from src.models.strategies.naivebayes_strategy import NaiveBayesStrategy


class NaiveBayesFactory(ModelFactory):
    def create_model(self, model_name: str, load_saved: bool) -> NaiveBayesStrategy:
        model =  NaiveBayesModel(model_name, load_saved).create_model()
        return NaiveBayesStrategy(model)