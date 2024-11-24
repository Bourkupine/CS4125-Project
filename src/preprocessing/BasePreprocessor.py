from src.preprocessing.Preprocessor import Preprocessor

from src.patterns.config_manager import ConfigManager
from pandas import DataFrame


class BasePreprocessor(Preprocessor):

    def __init__(self):
        self.config_manager = ConfigManager()

    def preprocess(self, df: DataFrame) -> DataFrame:
        ticket_summary = self.config_manager.get_config("TICKET_SUMMARY")
        interaction_content = self.config_manager.get_config("INTERACTION_CONTENT")

        # remove_empty
        df =  df.loc[(df[ticket_summary].notna()) & (df[interaction_content].notna())]

        df[interaction_content] = df[interaction_content].values.astype('U')
        df[ticket_summary] = df[ticket_summary].values.astype('U')

        return df
