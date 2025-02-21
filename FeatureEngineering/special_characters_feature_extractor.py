from Settings.keys import ParamsKeys
import pandas as pd

class SpecialCharactersFeatureExtractor:
    """Conta o total de caracteres especiais presentes na URL."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_nb_special_chars(self) -> pd.DataFrame:
        """Extrai a contagem de caracteres especiais da URL com os nomes das colunas padronizados."""

        char_mapping = {
            '.': 'nb_dots',
            '-': 'nb_hyphens',
            '@': 'nb_at',
            '?': 'nb_qm',
            '&': 'nb_and',
            '|': 'nb_or',
            '=': 'nb_eq',
            '_': 'nb_underscore',
            '~': 'nb_tilde',
            '%': 'nb_percent',
            '/': 'nb_slash',
            '*': 'nb_star',
            ':': 'nb_colon',
            ',': 'nb_comma',
            ';': 'nb_semicolumn',
            '$': 'nb_dollar',
            ' ': 'nb_space',
            'www': 'nb_www',
            '.com': 'nb_com'
        }

        # Contagem de caracteres individuais
        for char, col_name in char_mapping.items():
            if char in ['www', '.com']:
                self.df[col_name] = self.df['url'].apply(lambda x: x.lower().count(char))
            else:
                self.df[col_name] = self.df['url'].apply(lambda x: x.count(char))

        return self.df