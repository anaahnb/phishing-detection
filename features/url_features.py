from settings.keys import ParamsKeys
from urllib.parse import urlparse
import pandas as pd
import re

def extract_url_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai o comprimento das partes da URL."""

    df[ParamsKeys.URL_LENGTH] = df[ParamsKeys.URL].apply(len)

    df["hostname_length"] = df[ParamsKeys.URL].apply(lambda x: len(urlparse(x).hostname) if urlparse(x).hostname else 0)
    df["path_length"] = df[ParamsKeys.URL].apply(lambda x: len(urlparse(x).path))
    df["query_length"] = df[ParamsKeys.URL].apply(lambda x: len(urlparse(x).query))

    return df

def extract_special_characters_features(df: pd.DataFrame) -> pd.DataFrame:
    """Conta a ocorrência de caracteres especiais na URL."""

    special_chars = ['@', '-', '_', '/', '?', '=', '.', '&', '!', '~', ',', '+', '*', '#', '$', '%']

    for char in special_chars:
        df[f'qty_{char}_url'] = df[ParamsKeys.URL].apply(lambda x: x.count(char))

    return df

def extract_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica a presença de emails em partes específicas da URL."""

    df[ParamsKeys.EMAIL_IN_URL] = df[ParamsKeys.URL].apply(lambda x: 1 if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", x) else 0)
    df[ParamsKeys.EMAIL_IN_HOSTNAME] = df[ParamsKeys.URL].apply(lambda x: 1 if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", urlparse(x).hostname) else 0)
    df[ParamsKeys.EMAIL_IN_PATH] = df[ParamsKeys.URL].apply(lambda x: 1 if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", urlparse(x).path) else 0)

    return df

def extract_url_shortening_features(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica se a URL foi encurtada."""

    df[ParamsKeys.IS_SHORTENED_URL] = df[ParamsKeys.URL].apply(lambda x: 1 if re.match(r"(bit\.ly|goo\.gl|tinyurl\.com|is\.gd|buff\.ly)", x) else 0)

    return df

def extract_all_url_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai todas as features detalhadas das URLs."""

    df = extract_url_length_features(df)
    df = extract_special_characters_features(df)
    df = extract_email_features(df)
    df = extract_url_shortening_features(df)

    print("Extração de todas as features das URLs concluída.")

    return df
