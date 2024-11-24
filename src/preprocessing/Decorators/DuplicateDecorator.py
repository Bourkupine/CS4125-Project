from src.preprocessing.Preprocessor import Preprocessor
from pandas import DataFrame

class DuplicateDecorator(Preprocessor):

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def preprocess(self, df: DataFrame) -> DataFrame:
        #removing duplicates
        df = self.preprocessor.preprocess(df)
        return df.drop_duplicates(keep='first')
