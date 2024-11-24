from src.models.classifiers.RandomForest import RandomForestModel
from src.models.factories.ModelFactory import ModelFactory
from src.models.strategies.randomforest_strategy import RandomForestStrategy


class RandomForestFactory(ModelFactory):
    def create_model(self, model_name: str, load_saved: bool) -> RandomForestStrategy:
        model = RandomForestModel(model_name, load_saved)
        return RandomForestStrategy(model)