from turtle import pd
from pandas import DataFrame, read_csv
from src.config.config import Config
from src.data.preprocessing import PreProcessing
from src.utils.translate import trans_to_en

class TranslateDecorator(PreProcessing):

    def preprocess(self,df: DataFrame):
        #translating using trans_to_en
        df[Config.TICKET_SUMMARY] = pd.Series(trans_to_en(df[Config.TICKET_SUMMARY].to_list()), index=df.index)
        df[Config.INTERACTION_CONTENT] = pd.Series(trans_to_en(df[Config.INTERACTION_CONTENT].to_list()),
                                                   index=df.index)
        return df
