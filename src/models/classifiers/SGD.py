from sklearn.base import BaseEstimator

from src.models.BaseModel import BaseModel
from sklearn.linear_model import SGDClassifier

class SGD(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)

    def create_model(self) -> BaseEstimator:
        return SGDClassifier(loss='log_loss')
