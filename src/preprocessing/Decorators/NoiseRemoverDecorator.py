from pandas import DataFrame
from src.patterns.config_manager import ConfigManager
from src.preprocessing.Preprocessor import Preprocessor


class NoiseRemoverDecorator(Preprocessor):

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.config_manager = ConfigManager()

    def preprocess(self, df: DataFrame):
        df = self.preprocessor.preprocess(df)

        ticket_summary = self.config_manager.get_config("TICKET_SUMMARY")
        interaction_content = self.config_manager.get_config("INTERACTION_CONTENT")

        # removing noise
        noise = "(sv\\s*:)|(wg\\s*:)|(ynt\\s*:)|(fw(d)?\\s*:)|(r\\s*:)|(re\\s*:)|(\\[|\\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
        df[ticket_summary] = df[ticket_summary].str.lower().replace(noise, " ", regex=True).replace(
            r'\\s+', ' ', regex=True).str.strip()

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
            df[interaction_content] = df[interaction_content].str.lower().replace(noise, " ",
                                                                                regex=True).replace(
                r'\\s+', ' ', regex=True).str.strip()

        df = df.loc[(df[ticket_summary] != "") & (df[interaction_content] != "")]

        return df