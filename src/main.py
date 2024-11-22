import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelling.data_model import Data
from modelling.modelling import model_predict
from models import model
from src.config.config import Config
from src.data.embeddings import get_tfidf_embd
from src.data.preprocessing import de_duplication, noise_remover, get_input_data, remove_empty

seed = 0
random.seed(seed)
np.random.seed(seed)

def preprocess_data(df):
    df = remove_empty(df)
    df = de_duplication(df)
    noise_remover(df)
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, model: model):
    model_predict(data, df, model)

if __name__ == '__main__':
    df = get_input_data()

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    df = preprocess_data(df)

    X = get_embeddings(df)
    Y = df["Type 2"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    data = get_data_object(X, df)
    perform_modelling(data, df, 'name')
