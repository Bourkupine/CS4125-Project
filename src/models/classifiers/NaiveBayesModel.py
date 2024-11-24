from sklearn.naive_bayes import MultinomialNB

from src.models.classifiers.BaseModel import BaseModel


class NaiveBayesModel(BaseModel):
    def __init__(self, name: str, load_model: bool = False):
        super().__init__(name, load_model)

    def create_model(self):
        if self.load:
            model = self.load_model()
        else:
            model = MultinomialNB()
        return model if model else MultinomialNB()
