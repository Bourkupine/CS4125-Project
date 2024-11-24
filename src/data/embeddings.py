from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import concatenate

import sys
import os

# Needed so we can view the config manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patterns.config_manager import ConfigManager

def get_tfidf_embd(df: DataFrame):


    config_manager = ConfigManager()
    interaction_content = config_manager.get_config("INTERACTION_CONTENT")
    ticket_summary = config_manager.get_config("TICKET_SUMMARY")

    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    x1 = tfidfconverter.fit_transform(df[interaction_content]).toarray()
    x2 = tfidfconverter.fit_transform(df[ticket_summary]).toarray()
    X = concatenate((x1, x2), axis=1)
    return X