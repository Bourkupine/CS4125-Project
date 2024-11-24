from sklearn.linear_model import SGDClassifier

from src.models.classifiers.BaseModel import BaseModel


class SGDModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        self.model = self.load_model() if self.load else SGDClassifier(loss='log_loss')
