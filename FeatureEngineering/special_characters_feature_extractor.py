from Settings.keys import ParamsKeys
import pandas as pd

class SpecialCharactersFeatureExtractor:
    """Conta o total de caracteres especiais presentes na URL."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        special_chars = ['@', '-', '_', '/', '?', '=', '.', '&', '!', '~', ',', '+', '*', '#', '$', '%']
        self.df[ParamsKeys.SPECIAL_CHAR_COUNT] = self.df[ParamsKeys.URL].apply(lambda x: sum(x.count(char) for char in special_chars))
        return self.df
