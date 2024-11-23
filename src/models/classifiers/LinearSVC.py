from sklearn.base import BaseEstimator

from src.models.BaseModel import BaseModel
from sklearn.svm import LinearSVC

class LinearSVModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)

    def create_model(self) -> BaseEstimator:
        return LinearSVC()