from sklearn.ensemble import AdaBoostClassifier

from src.models.classifiers.BaseModel import BaseModel


class AdaBoostModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        self.model = self.load_model() if self.load else AdaBoostClassifier(n_estimators=1000)
