from pandas import DataFrame, read_csv

def get_input_data(self) -> DataFrame:
    return read_csv("../datasets/AppGallery.csv")
