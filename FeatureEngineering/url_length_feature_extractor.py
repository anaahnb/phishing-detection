from Settings.keys import ParamsKeys
from urllib.parse import urlparse
import pandas as pd

class UrlLengthFeatureExtractor:
    """Extrai o comprimento das partes da URL."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        self.df[ParamsKeys.URL_LENGTH] = self.df[ParamsKeys.URL].apply(len)
        self.df[ParamsKeys.HOSTNAME_LENGTH] = self.df[ParamsKeys.URL].apply(lambda x: len(urlparse(x).hostname) if urlparse(x).hostname else 0)
        self.df[ParamsKeys.PATH_LENGTH] = self.df[ParamsKeys.URL].apply(lambda x: len(urlparse(x).path))
        self.df[ParamsKeys.QUERY_LENGTH] = self.df[ParamsKeys.URL].apply(lambda x: len(urlparse(x).query))
        return self.df
