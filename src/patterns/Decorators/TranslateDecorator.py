from turtle import pd
from pandas import DataFrame, read_csv
from src.patterns.config_manager import ConfigManager
from src.data.Preprocessor import Preprocessor
from src.utils.translate import trans_to_en

class TranslateDecorator(Preprocessor):

    def __init__(self, PreProcessing):
        self.PreProcessing = PreProcessing
        self.config_manager = ConfigManager()

    def preprocess(self,df: DataFrame):
        #translating using trans_to_en
        self.PreProcessing.preprocess()

        ticket_summary = self.config_manager.get_config("TICKET_SUMMARY")
        interaction_content = self.config_manager.get_config("INTERACTION_CONTENT")
        df[ticket_summary] = pd.Series(trans_to_en(df[ticket_summary].to_list()), index=df.index)
        df[interaction_content] = pd.Series(trans_to_en(df[interaction_content].to_list()),
                                                   index=df.index)
        return df
