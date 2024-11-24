from pandas import DataFrame
from abc import ABC, abstractmethod

class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, text:DataFrame) -> DataFrame:
        pass
