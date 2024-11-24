from sklearn.ensemble import RandomForestClassifier

from src.models.classifiers.BaseModel import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        self.model = self.load_model() if self.load else RandomForestClassifier(n_estimators=1000,class_weight='balanced_subsample')
