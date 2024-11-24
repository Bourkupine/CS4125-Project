from src.models.classifiers.LinearSVC import LinearSVCModel
from src.models.factories.ModelFactory import ModelFactory
from src.models.strategies.linearSVC_strategy import LinearSVCStrategy


class LinearSVCFactory(ModelFactory):
    def create_model(self, model_name: str, load_saved: bool) -> LinearSVCStrategy:
        model =  LinearSVCModel(model_name, load_saved).create_model()
        return LinearSVCStrategy(model)