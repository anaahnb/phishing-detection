from Settings.keys import ParamsKeys
import tldextract
import pandas as pd
from urllib.parse import urlparse

class TLDExtract:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_tld_features(self) -> pd.DataFrame:
        self.df['tld_in_path'] = self.df['url'].apply(lambda x: 1 if tldextract.extract(x).suffix in urlparse(x).path else 0)
        self.df['tld_in_subdomain'] = self.df['url'].apply(lambda x: 1 if tldextract.extract(x).suffix in tldextract.extract(x).subdomain else 0)
        return self.df
