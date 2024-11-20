import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import *
from src.utils import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data():

    # This method will create the model for data
    # This will be performed in second activity
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        self.df = df
        self.X = X
        ...

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df