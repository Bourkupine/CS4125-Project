from sklearn.base import BaseEstimator

from src.models.BaseModel import BaseModel
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes(BaseModel):
    def __init__(self, name: str):
        super().__init__(name)

    def create_model(self) -> BaseEstimator:
        return MultinomialNB()