from sklearn.ensemble import AdaBoostClassifier

from src.models.classifiers.BaseModel import BaseModel


class AdaBoostModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        if self.load:
            model = self.load_model()
        else:
            model = AdaBoostClassifier(n_estimators=1000)

        return model if model else AdaBoostClassifier(n_estimators=1000)