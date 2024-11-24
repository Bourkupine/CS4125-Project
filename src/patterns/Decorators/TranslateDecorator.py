from turtle import pd
from pandas import DataFrame, read_csv
from src.config.config import Config
from src.data.Preprocessor import Preprocessor
from src.utils.translate import trans_to_en

class TranslateDecorator(Preprocessor):

    def __init__(self, PreProcessing):
        self.PreProcessing = PreProcessing

    def preprocess(self,df: DataFrame):
        #translating using trans_to_en
        self.PreProcessing.preprocess()
        df[Config.TICKET_SUMMARY] = pd.Series(trans_to_en(df[Config.TICKET_SUMMARY].to_list()), index=df.index)
        df[Config.INTERACTION_CONTENT] = pd.Series(trans_to_en(df[Config.INTERACTION_CONTENT].to_list()),
                                                   index=df.index)
        return df
