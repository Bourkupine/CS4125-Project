from sklearn.linear_model import LogisticRegression

from src.models.classifiers.BaseModel import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        self.model = self.load_model() if self.load else LogisticRegression(max_iter=1000)
