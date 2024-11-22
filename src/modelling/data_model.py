import random

import numpy as np

class Data():

    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray) -> None:
        self.Y_test = Y_test
        self.y_train = Y_train
        self.X_test = X_test
        self.X_train = X_train

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.Y_test

