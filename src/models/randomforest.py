from sklearn.base import BaseEstimator

from src.models.Model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from numpy import *
import random

seed = 0
np.random.seed(seed)
random.seed(seed)

class RandomForest(BaseModel):
    def __init__(self) -> None:
        super(RandomForest, self).__init__()

    def create_model(self) -> BaseEstimator:
        return RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')

