from preprocessing import PreProcessing
from src.config.config import Config
from pandas import DataFrame


class BasePreProcessing(PreProcessing):

    def preprocess(self,df:DataFrame) -> DataFrame:
        #remove_empty
        return df.loc[(df[Config.TICKET_SUMMARY].notna()) & (df[Config.INTERACTION_CONTENT].notna())]
