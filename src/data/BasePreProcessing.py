from preprocessing import PreProcessing
from src.config.config import Config
from pandas import DataFrame, read_csv


class BasePreProcessing(PreProcessing):

    def preprocess(df) -> DataFrame:
        #remove_empty
        return df.loc[(df[Config.TICKET_SUMMARY].notna()) & (df[Config.INTERACTION_CONTENT].notna())]

    def get_input_data(self) -> DataFrame:
        return read_csv("../datasets/AppGallery.csv")
