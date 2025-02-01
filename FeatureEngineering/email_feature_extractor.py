from Settings.keys import ParamsKeys
import pandas as pd
import re
from urllib.parse import urlparse

class EmailFeatureExtractor:
    """Verifica a presença de emails nas partes específicas da URL."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        self.df[ParamsKeys.EMAIL_IN_URL] = self.df[ParamsKeys.URL].apply(lambda x: 1 if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", x) else 0)
        self.df[ParamsKeys.EMAIL_IN_HOSTNAME] = self.df[ParamsKeys.URL].apply(lambda x: 1 if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA0-9-.]+", urlparse(x).hostname) else 0)
        self.df[ParamsKeys.EMAIL_IN_PATH] = self.df[ParamsKeys.URL].apply(lambda x: 1 if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", urlparse(x).path) else 0)
        return self.df
