from sklearn.ensemble import RandomForestClassifier

from src.models.classifiers.BaseModel import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        if self.load:
            model = self.load_model()
        else:
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
        return model if model else RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample')
