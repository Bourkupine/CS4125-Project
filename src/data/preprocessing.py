import pandas as pd
from pandas import DataFrame, read_csv

from src.config.config import Config
from src.utils.translate import trans_to_en


def remove_empty(df) -> DataFrame:
    return df.loc[(df[Config.TICKET_SUMMARY].notna()) & (df[Config.INTERACTION_CONTENT].notna())]

def de_duplication(df: DataFrame) -> DataFrame:
    return df.drop_duplicates(keep='first')


def noise_remover(df: DataFrame):
    noise = "(sv\\s*:)|(wg\\s*:)|(ynt\\s*:)|(fw(d)?\\s*:)|(r\\s*:)|(re\\s*:)|(\\[|\\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower().replace(noise, " ", regex=True).replace(r'\\s+',' ',regex=True).str.strip()

    noise_1 = [
        "(from :)|(subject :)|(sent :)|(r\\s*:)|(re\\s*:)",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        "\\d{2}(:|.)\\d{2}",
        "(xxxxx@xxxx\\.com)|(\\*{5}\\([a-z]+\\))",
        "dear ((customer)|(user))",
        "dear",
        "(hello)|(hallo)|(hi )|(hi there)",
        "good morning",
        "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        "thank you for contacting us",
        "thank you for your availability",
        "thank you for providing us this information",
        "thank you for contacting",
        "thank you for reaching us (back)?",
        "thank you for patience",
        "thank you for (your)? reply",
        "thank you for (your)? response",
        "thank you for (your)? cooperation",
        "thank you for providing us with more information",
        "thank you very kindly",
        "thank you( very much)?",
        "i would like to follow up on the case you raised on the date",
        "i will do my very best to assist you"
        "in order to give you the best solution",
        "could you please clarify your request with following information:"
        "in this matter",
        "we hope you(( are)|('re)) doing ((fine)|(well))",
        "i would like to follow up on the case you raised on",
        "we apologize for the inconvenience",
        "sent from my huawei (cell )?phone",
        "original message",
        "customer support team",
        "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        "canada, australia, new zealand and other countries",
        "\\d+",
        "[^0-9a-zA-Z]+",
        "(\\s|^).(\\s|$)"]

    for noise in noise_1:
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower().replace(noise, " ", regex=True).replace(r'\\s+',' ',regex=True).str.strip()

def get_input_data() -> DataFrame:
    return read_csv("../datasets/AppGallery.csv")


def translate_input_data(df: DataFrame):
    df[Config.TICKET_SUMMARY] = pd.Series(trans_to_en(df[Config.TICKET_SUMMARY].to_list()), index=df.index)
    # df[Config.INTERACTION_CONTENT] = pd.Series(trans_to_en(df[Config.INTERACTION_CONTENT].to_list()), index=df.index)
