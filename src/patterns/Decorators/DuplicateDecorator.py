from src.data.preprocessing import PreProcessing
from pandas import DataFrame

class DuplicateDecorator(PreProcessing):

    def preprocess(self,df: DataFrame) -> DataFrame:
        #removing duplicates
        return df.drop_duplicates(keep='first')

    def __init__(self,df: DataFrame)-> DataFrame:
        self.preprocess()
