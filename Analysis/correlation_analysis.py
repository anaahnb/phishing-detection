import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DataLoading.data_loader import DataLoader

class CorrelationAnalysis:
    """Classe responsável por gerar e salvar a matriz de correlação das variáveis numéricas."""

    def _init_(self, df: pd.DataFrame, output_dir: str = "Figures"):
        self.df = df.select_dtypes(include=[np.number])
        self.output_dir = output_dir
        self._prepare_output_directory()

    def _prepare_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_correlation_matrix(self):
        """Gera e salva a matriz de correlação das variáveis numéricas."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
        plt.title('Matriz de Correlação das Variáveis')
        plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight")
        print(f'A matriz foi salva em {self.output_dir}')

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data()

    correlation_analysis = CorrelationAnalysis(df)
    correlation_analysis.plot_correlation_matrix()

