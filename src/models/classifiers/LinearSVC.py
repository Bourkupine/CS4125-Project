from sklearn.svm import LinearSVC

from src.models.classifiers.BaseModel import BaseModel


class LinearSVCModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        if self.load:
            model = self.load_model()
        else:
            model = LinearSVC()

        return model if model else LinearSVC()
