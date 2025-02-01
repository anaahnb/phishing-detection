from FeatureEngineering import ( UrlLengthFeatureExtractor, SpecialCharactersFeatureExtractor, EmailFeatureExtractor, UrlShorteningFeatureExtractor)
import pandas as pd

class UrlFeatureExtractor:
    """Classe principal para coordenar a extração de todas as features das URLs."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_all(self) -> pd.DataFrame:
        # Extrair as features de comprimento
        url_length_extractor = UrlLengthFeatureExtractor(self.df)
        self.df = url_length_extractor.extract()

        # Extrair as features de caracteres especiais
        special_chars_extractor = SpecialCharactersFeatureExtractor(self.df)
        self.df = special_chars_extractor.extract()

        # Extrair as features de email
        email_extractor = EmailFeatureExtractor(self.df)
        self.df = email_extractor.extract()

        # Extrair as features de encurtamento
        url_shortening_extractor = UrlShorteningFeatureExtractor(self.df)
        self.df = url_shortening_extractor.extract()

        print("Extração de todas as features das URLs concluída.")
        return self.df
