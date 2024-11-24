from abc import abstractmethod

from src.models.strategies.model_strategy import ModelStrategy


class ModelFactory:
    @abstractmethod
    def create_model(self, model_name: str, load_saved: bool) -> ModelStrategy:
        ...
