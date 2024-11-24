from src.data.Preprocessor import Preprocessor
from pandas import DataFrame

class DuplicateDecorator(Preprocessor):

    def __init__(self,PreProcessing):
        self.PreProcessing = PreProcessing

    def preprocess(self,df: DataFrame) -> DataFrame:
        #removing duplicates
        self.PreProcessing.preprocess()
        return df.drop_duplicates(keep='first')
