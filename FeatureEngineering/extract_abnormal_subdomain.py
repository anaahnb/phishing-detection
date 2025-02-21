from Settings.keys import ParamsKeys
import tldextract
import pandas as pd

class AbnormalSubdomainExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_abnormal_subdomain(self) -> pd.DataFrame:
        self.df['nb_subdomains'] = self.df['url'].apply(lambda x: len(tldextract.extract(x).subdomain.split('.')))
        self.df['abnormal_subdomain'] = self.df['nb_subdomains'].apply(lambda x: 1 if x > 2 else 0)
        return self.df