from src.data.Preprocessor import Preprocessor
import sys
import os
# Needed so we can view the config manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from patterns.config_manager import ConfigManager
from pandas import DataFrame


class BasePreprocessor(Preprocessor):

    def __init__(self):
        self.config_manager = ConfigManager

    def preprocess(self,df:DataFrame) -> DataFrame:
        ticket_summary = self.config_manager.get_config("TICKET_SUMMARY")
        interaction_content = self.config_manager.get_config("INTERACTION_CONTENT")

        #remove_empty
        return df.loc[(df[ticket_summary].notna()) & (df[interaction_content].notna())]
