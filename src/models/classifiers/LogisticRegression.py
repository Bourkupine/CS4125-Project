from sklearn.base import BaseEstimator

from src.models.BaseModel import BaseModel
from sklearn.linear_model import LogisticRegression

class RandomForest(BaseModel):
    def __init__(self, name: str):
        super(RandomForest, self).__init__(name)

    def create_model(self) -> BaseEstimator:
        return LogisticRegression(max_iter=1000)

