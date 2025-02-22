import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from DataLoading import DataDownloader, DataExtractor
from Settings.keys import ParamsKeys

class DataLoader:
    def __init__(self, filepath: str = None):
        self.filepath = filepath

    def get_dataset_path(self, base_path: str) -> str:
        """Determina o caminho correto do dataset, verificando se é um ZIP ou CSV."""

        zip_file = os.path.join(base_path, ParamsKeys.ZIP_FILE_NAME)
        extracted_folder = os.path.join(base_path, ParamsKeys.EXTRACTED_FOLDER_NAME)

        if os.path.exists(zip_file):
            return DataExtractor.extract(zip_file, extracted_folder)

        csv_file = os.path.join(base_path, ParamsKeys.CSV_FILE_NAME)
        if os.path.exists(csv_file):
            return base_path

        raise FileNotFoundError(f"Arquivo CSV não encontrado em: {base_path}")

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados a partir de um arquivo CSV. Se não for fornecido, baixa e processa os dados."""

        if self.filepath is None:
            downloader = DataDownloader()
            base_path = downloader.download()
            self.filepath = self.get_dataset_path(base_path)

        csv_file = os.path.join(self.filepath, ParamsKeys.CSV_FILE_NAME)

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"O arquivo CSV não foi encontrado em: {csv_file}")

        return pd.read_csv(csv_file)

if __name__ == "__main__":
    data_loader = DataLoader()
    df = data_loader.load_data()

    if df is not None and not df.empty:
        print(f"Dados carregados com sucesso! O dataset contém {df.shape[0]} linhas e {df.shape[1]} colunas.")
    else:
        print("Falha ao carregar os dados.")
