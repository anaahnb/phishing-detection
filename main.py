import os
from DataLoading.data_loader import DataLoader
from DataEngineering.data_cleaner import DataCleaner
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

class DataPreprocessing:
    def _init_(self, output_dir="Datasets/processed"):
        self.output_dir = output_dir
        self.df = None

    def load_data(self):
        """Carrega os dados usando a classe DataLoader."""
        data_loader = DataLoader()
        self.df = data_loader.load_data()

    def clean_data(self):
        """Aplica limpeza de dados usando a classe DataCleaner."""
        data_cleaner = DataCleaner(self.df)
        self.df = data_cleaner.clean()

    def extract_features(self):
        """Extrai as features utilizando a classe UrlFeatureExtractor."""
        feature_extractor = UrlFeatureExtractor(self.df)
        self.df = feature_extractor.extract_all()

    def save_data(self):
        """Salva os dados pré-processados no diretório especificado."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_path = os.path.join(self.output_dir, "phishing_dataset_cleaned.csv")
        self.df.to_csv(file_path, index=False)
        print(f"Pré-processamento concluído! Dados salvos em {file_path}")

    def run(self):
        """Executa todas as etapas do pré-processamento."""
        self.load_data()
        self.clean_data()
        self.extract_features()
        self.save_data()

if _name_ == "_main_":
    preprocessor = DataPreprocessing()
    preprocessor.run()