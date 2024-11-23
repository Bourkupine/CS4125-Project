import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from src.data.preprocessing import de_duplication, noise_remover, get_input_data, remove_empty, translate_input_data
from src.config.config import Config
from src.utils.translate import trans_to_en
from src.data.embeddings import get_tfidf_embd
from modelling.data_model import Data
from patterns.observer import ClassificationNotifier, MetricsTracker, UIUpdater
from utils.logger import log_classification
from src.data.BasePreProcessing import BasePreProcessing
from patterns.Decorators import DuplicateDecorator, NoiseRemoverDecorator, TranslateDecorator
import random

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
#this is to initilise the basepreprocessing class so we can use it here.
BasePreProcessor = BasePreProcessing()
DuplicateDecorator = DuplicateDecorator()
TranslateDecorator = TranslateDecorator()
NoiseRemoverDecorator = NoiseRemoverDecorator()


def load_data():
    # load the input data
    df = BasePreProcessor.get_input_data()
    return df


def preprocess_data(df):
    #Preprocess the input data.
    df = BasePreProcessor.preprocess(df)
    df = DuplicateDecorator.preprocess_data(df)
    df = TranslateDecorator.preprocess(df)
    df = NoiseRemoverDecorator.preprocess(df)
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def split_data(X, df):
    #Split the data into training and testing sets.
    Y = df["Type 2"].to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    return Data(X_train, X_test, Y_train, Y_test)


def perform_modelling(data, df, name):
    #Perform modeling on the given data.
    # Placeholder for actual model implementation
    pass


def classify_email(email, notifier):

    #Classify an email and notify observers.

    #Args:email (str): The email content to classify. notifier (ClassificationNotifier): The notifier instance managing observers.

    # Placeholder for classification logic
    classification = "Spam" if "offer" in email.lower() else "Not Spam"

    # Log the classification
    log_classification(email, classification)

    # Notify observers about the classification
    notifier.notify(email, classification)


# Code execution starts here
if __name__ == '__main__':
    # Initialize notifier and attach observers
    notifier = ClassificationNotifier()
    notifier.attach(MetricsTracker())
    notifier.attach(UIUpdater())

    # Load and preprocess data
    df = load_data()
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    df = preprocess_data(df)

    # Generate embeddings and split data
    X, processed_df = get_embeddings(df)
    data = split_data(X, processed_df)

    # Perform modeling (placeholder for actual implementation)
    perform_modelling(data, processed_df, "name")

    # Classify emails and notify observers
    for email in df[Config.INTERACTION_CONTENT]:
        classify_email(email, notifier)
