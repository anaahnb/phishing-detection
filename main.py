import os
from data.data_loader import load_data
from data.data_processing import clean_data

df = load_data()
df = clean_data(df)

output_dir = "data/processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(f"{output_dir}/phishing_dataset_cleaned.csv", index=False)
print("Pré-processamento concluído! Dados salvos em data/processed")