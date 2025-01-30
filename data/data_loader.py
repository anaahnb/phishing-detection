import os
import pandas as pd
import kagglehub
import zipfile

def download_data():
    """Baixa a versão mais recente do dataset diretamente do Kaggle."""
    path = kagglehub.dataset_download("shashwatwork/web-page-phishing-detection-dataset")
    print(f"Arquivos baixados para: {path}")

    # Caso o arquivo seja um ZIP, será extraído
    zip_file = f"{path}/web_page_phishing_detection.zip"
    extracted_folder = f"{path}/extracted/"

    # Caso o arquivo ZIP não seja encontrado, verifique se é um CSV diretamente
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)
        print(f"Arquivo extraído para: {extracted_folder}")
        return extracted_folder
    else:
        csv_file = f"{path}/dataset_phishing.csv"
        if os.path.exists(csv_file):
            return path
        else:
            raise FileNotFoundError(f"Arquivo CSV não encontrado em: {path}")

def load_data(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        filepath = download_data()

    csv_file = f"{filepath}/dataset_phishing.csv"

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"O arquivo CSV não foi encontrado em: {csv_file}")

    df = pd.read_csv(csv_file)

    return df
