from sklearn.base import BaseEstimator

from src.models.BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseModel):
    def __init__(self, name: str):
        super(RandomForest, self).__init__(name)

    def create_model(self) -> BaseEstimator:
        return RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample')

