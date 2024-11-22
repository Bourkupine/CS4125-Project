from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import concatenate
from src.config.config import Config


def get_tfidf_embd(df: DataFrame):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidfconverter.fit_transform(df[Config.INTERACTION_CONTENT]).toarray()
    x2 = tfidfconverter.fit_transform(df[Config.TICKET_SUMMARY]).toarray()
    X = concatenate((x1, x2), axis=1)
    return X