from Settings.keys import ParamsKeys
import pandas as pd
from sklearn.model_selection import train_test_split

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_null_values(self):
        """Remove valores nulos do DataFrame."""
        self.df = self.df.dropna()

    def transform_status_column(self):
        """Transforma a coluna status de string para inteiro (1 para phishing, 0 para legítimo)."""
        self.df[ParamsKeys.STATUS] = self.df[ParamsKeys.STATUS].map({'phishing': 1, 'legitimate': 0})

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Divide os dados de forma estratificada para preservar a distribuição dos outliers."""

        # Identificando os outliers
        q1_url = self.df[ParamsKeys.URL_LENGTH].quantile(0.25)
        q3_url = self.df[ParamsKeys.URL_LENGTH].quantile(0.75)
        iqr_url = q3_url - q1_url
        lower_bound_url = q1_url - 1.5 * iqr_url
        upper_bound_url = q3_url + 1.5 * iqr_url

        q1_host = self.df[ParamsKeys.HOSTNAME_LENGTH].quantile(0.25)
        q3_host = self.df[ParamsKeys.HOSTNAME_LENGTH].quantile(0.75)
        iqr_host = q3_host - q1_host
        lower_bound_host = q1_host - 1.5 * iqr_host
        upper_bound_host = q3_host + 1.5 * iqr_host

        df_outliers = self.df[(self.df[ParamsKeys.URL_LENGTH] > upper_bound_url) |
                              (self.df[ParamsKeys.URL_LENGTH] < lower_bound_url) |
                              (self.df[ParamsKeys.HOSTNAME_LENGTH] > upper_bound_host) |
                              (self.df[ParamsKeys.HOSTNAME_LENGTH] < lower_bound_host)]
        df_normal = self.df.drop(df_outliers.index)

        # Divisão da base normal
        train_normal, test_normal = train_test_split(df_normal, test_size=test_size, random_state=random_state)
        train_normal, val_normal = train_test_split(train_normal, test_size=val_size, random_state=random_state)

        # Divisão da base de outliers de forma proporcional
        train_outliers, test_outliers = train_test_split(df_outliers, test_size=test_size, random_state=random_state)
        train_outliers, val_outliers = train_test_split(train_outliers, test_size=val_size, random_state=random_state)

        # Reunindo os conjuntos
        train_set = pd.concat([train_normal, train_outliers])
        test_set = pd.concat([test_normal, test_outliers])
        val_set = pd.concat([val_normal, val_outliers])

        return train_set, test_set, val_set

    def clean_and_split(self):
        """Executa a limpeza dos dados e os divide."""
        self.remove_null_values()
        return self.split_data()
