from src.models.classifiers.SGD import SGDModel
from src.models.factories.ModelFactory import ModelFactory
from src.models.strategies.SGD_strategy import SGDStrategy


class SGDFactory(ModelFactory):
    def create_model(self, model_name: str, load_saved: bool) -> SGDStrategy:
        model = SGDModel(model_name, load_saved).create_model()
        return SGDStrategy(model)