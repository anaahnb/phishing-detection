from Settings.keys import ParamsKeys
import pandas as pd

class DataCleaner:
    """Classe responsável pela limpeza dos dados, removendo valores nulos e outliers"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_null_values(self):
        """Remove valores nulos do DataFrame."""
        self.df = self.df.dropna()

    def remove_outliers(self):
        """Remove outliers da variável de comprimento da URL usando o método do intervalo interquartil (IQR)"""
        q1 = self.df[ParamsKeys.URL_LENGTH].quantile(0.25)
        q3 = self.df[ParamsKeys.URL_LENGTH].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        self.df = self.df[(self.df[ParamsKeys.URL_LENGTH] >= lower_bound) & (self.df[ParamsKeys.URL_LENGTH] <= upper_bound)]

    def clean(self) -> pd.DataFrame:
        """Executa todas as etapas de limpeza de dados e retorna o DataFrame tratado"""
        self.remove_null_values()
        self.remove_outliers()
        return self.df
