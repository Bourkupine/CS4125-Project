import numpy as np
import pandas as pd

from src.data.preprocessing import de_duplication, noise_remover, get_input_data
from src.config.config import Config
from src.utils.translate import trans_to_en
from src.data.embeddings import get_tfidf_embd
from modelling.modelling import model_predict
from modelling.data_model import Data
from patterns.observer import EmailClassifier, Logger, MetricsTracker

import random

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # load the input data
    df = get_input_data()
    return df


def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = trans_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)


def classify_emails(email_classifier, df):
    # Simulate classifying email content and notify observers
    for email in df[Config.INTERACTION_CONTENT]:
        email_classifier.classify(email)


# Code will start executing from following line
if __name__ == '__main__':
    # Initialize EmailClassifier and attach observers
    email_classifier = EmailClassifier()
    email_classifier.attach(Logger())
    email_classifier.attach(MetricsTracker())

    # pre-processing steps
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Classify emails and notify observers
    classify_emails(email_classifier, df)

    # data transformation
    X, group_df = get_embeddings(df)
    # data modelling
    data = get_data_object(X, df)
    # modelling
    perform_modelling(data, df, 'name')
