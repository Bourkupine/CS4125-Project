from src.data.preprocessing import PreProcessing
from pandas import DataFrame

class DuplicateDecorator(PreProcessing):
    def preprocess(df: DataFrame) -> DataFrame:
        return df.drop_duplicates(keep='first')
