from Settings.keys import ParamsKeys
import pandas as pd
from urllib.parse import urlparse

class HttpsTokenExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_https_token(self) -> pd.DataFrame:
        self.df['https_token'] = self.df['url'].apply(lambda x: 1 if 'https' in urlparse(x).path else 0)
        return self.df
