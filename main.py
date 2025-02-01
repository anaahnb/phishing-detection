import os
from DataLoading import DataLoader
from DataEngineering import DataCleaner
from FeatureEngineering import UrlFeatureExtractor

data_loader = DataLoader()
df = data_loader.load_data()

data_cleaner = DataCleaner(df)
df = data_cleaner.clean()

output_dir = "Datasets/processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

features = UrlFeatureExtractor(df)
df = features.extract_all()

df.to_csv(f"{output_dir}/phishing_dataset_cleaned.csv", index=False)
print(f"Pré-processamento concluído! Dados salvos em {output_dir}")