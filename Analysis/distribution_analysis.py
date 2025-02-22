import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from DataLoading.data_loader import DataLoader
from Settings.keys import ParamsKeys

class DistributionAnalysis:
    """Realiza a análise de distribuição das variáveis do conjunto de dados."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "Figures"):
        self.df = df
        self.df_numeric = df.select_dtypes(include=[np.number])
        self.output_dir = output_dir
        self._ensure_output_directory()

    def _ensure_output_directory(self):
        """Garante que o diretório de saída exista."""
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_target_distribution(self):
        """Gera e salva o gráfico de distribuição da variável alvo."""
        plt.figure(figsize=(6, 4))
        sns.countplot(x=ParamsKeys.STATUS, data=self.df, palette='Set2')
        plt.title('Distribuição da Variável Alvo')
        plt.savefig(f"{self.output_dir}/target_distribution.png", dpi=300, bbox_inches="tight")

    def plot_variable_distribution(self, num_cols=6):
        """Gera e salva histogramas das variáveis numéricas."""
        selected_cols = self.df_numeric.columns[:num_cols]
        self.df_numeric[selected_cols].hist(bins=30, figsize=(12, 10), color='steelblue', edgecolor='black')
        plt.suptitle('Distribuição das Variáveis Numéricas')
        plt.savefig(f"{self.output_dir}/variable_distribution.png", dpi=300, bbox_inches="tight")

    def plot_boxplots(self, num_cols=6):
        """Gera e salva boxplots das variáveis numéricas."""
        selected_cols = self.df_numeric.columns[:num_cols]
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.df_numeric[selected_cols], orient='h', palette='Set2')
        plt.title('Boxplots das Variáveis Numéricas')
        plt.savefig(f"{self.output_dir}/boxplots.png", dpi=300, bbox_inches="tight")

    def run_analysis(self):
        """Executa toda a análise de distribuição e salva os gráficos."""
        self.plot_target_distribution()
        self.plot_variable_distribution()
        self.plot_boxplots()
        print(f'Gráficos salvos em {self.output_dir}')

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data()
    analysis = DistributionAnalysis(df)
    analysis.run_analysis()
