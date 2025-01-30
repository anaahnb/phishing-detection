from settings.keys import ParamsKeys
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza limpeza dos dados, removendo valores nulos e outliers."""

    df = df.dropna()

    q1 = df[ParamsKeys.URL_LENGTH].quantile(0.25)
    q3 = df[ParamsKeys.URL_LENGTH].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[ParamsKeys.URL_LENGTH] >= lower_bound) & (df[ParamsKeys.URL_LENGTH] <= upper_bound)]

    return df
