from Settings.keys import ParamsKeys
import pandas as pd

class SpecialCharactersFeatureExtractor:
    """Conta o total de caracteres especiais presentes na URL."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_nb_special_chars(self) -> pd.DataFrame:
        special_chars = ['.', '-', '@', '?', '&', '|', '=', '_', '~', '%', '/', '*', ':', ',', ';', '$', ' ']
        for char in special_chars:
            self.df[f'nb_{char}'] = self.df['url'].apply(lambda x: x.count(char))
        return self.df