from Settings.keys import ParamsKeys
import pandas as pd
import re

class UrlShorteningFeatureExtractor:
    """Verifica se a URL foi encurtada."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        self.df[ParamsKeys.IS_SHORTENED_URL] = self.df[ParamsKeys.URL].apply(lambda x: 1 if re.match(r"(bit\.ly|goo\.gl|tinyurl\.com|is\.gd|buff\.ly)", x) else 0)
        return self.df
