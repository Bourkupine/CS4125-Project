from pandas import DataFrame, read_csv

def get_input_data() -> DataFrame:
    return read_csv("datasets/AppGallery.csv")
