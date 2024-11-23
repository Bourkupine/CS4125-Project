from pandas import DataFrame
from abc import ABC, abstractmethod

class PreProcessing(ABC):
    @abstractmethod
    def preprocess(self, text:DataFrame) -> DataFrame:
        pass
