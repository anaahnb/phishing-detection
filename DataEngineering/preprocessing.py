import pandas as pd
from sklearn.model_selection import train_test_split
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor
from Settings.keys import ParamsKeys

class DataPreprocessor:
    """Classe responsável pelo pré-processamento dos dados, incluindo extração de features e divisão do conjunto de treino e teste."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_features(self):
        """Extrai as features do dataset utilizando a classe de extração de features."""
        extractor = UrlFeatureExtractor(self.df)
        self.df = extractor.extract_all()

    def remove_categorical_features(self):
        """Remove colunas categóricas, exceto a variável alvo (STATUS)."""
        for col in self.df.columns:
            if self.df[col].dtype == "object" and col != ParamsKeys.STATUS:
                self.df = self.df.drop(columns=[col])

    def split_data(self, test_size=0.34, random_state=42):
        """Divide os dados em treino e teste utilizando o padrão 66% treino / 34% teste."""
        y = self.df[ParamsKeys.STATUS]
        X = self.df.drop(columns=[ParamsKeys.STATUS])
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def encode_labels(self):
      """Converte rótulos 'legitimate' e 'phishing' para valores binários."""
      self.df[ParamsKeys.STATUS] = self.df[ParamsKeys.STATUS].map({"legitimate": 0, "phishing": 1})

    def preprocess(self):
        """Executa todas as etapas do pré-processamento e retorna os conjuntos de treino e teste."""
        self.extract_features()
        self.remove_categorical_features()
        self.encode_labels()
        return self.split_data()
