from Settings.keys import ParamsKeys
import pandas as pd
import re

class IPExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_ip(self) -> pd.DataFrame:
        ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.df['ip'] = self.df['url'].apply(lambda x: 1 if ip_pattern.search(x) else 0)
        return self.df
