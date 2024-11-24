from pandas import DataFrame, Series
from src.patterns.config_manager import ConfigManager
from src.preprocessing.Preprocessor import Preprocessor
from src.utils.translate import trans_to_en


class TranslateDecorator(Preprocessor):

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.config_manager = ConfigManager()

    def preprocess(self, df: DataFrame):
        # translating using trans_to_en
        df = self.preprocessor.preprocess(df)

        ticket_summary = self.config_manager.get_config("TICKET_SUMMARY")
        interaction_content = self.config_manager.get_config("INTERACTION_CONTENT")
        df[ticket_summary] = Series(trans_to_en(df[ticket_summary].to_list()), index=df.index)
        df[interaction_content] = Series(trans_to_en(df[interaction_content].to_list()),
                                         index=df.index)
        return df
