import os
import shutil
import kagglehub
from Settings.keys import ParamsKeys

class DataDownloader:
    @staticmethod
    def download() -> str:
        """Baixa a versão mais recente do dataset do Kaggle e move para a pasta correta."""

        cache_path = kagglehub.dataset_download(ParamsKeys.DATASET_NAME)
        print(f"Arquivos baixados para: {cache_path}")

        datasets_path = os.path.join(os.path.dirname(__file__), "..", "Datasets", "raw")
        os.makedirs(datasets_path, exist_ok=True)  # Garante que a pasta existe

        for filename in os.listdir(cache_path):
            src = os.path.join(cache_path, filename)
            dst = os.path.join(datasets_path, filename)
            shutil.move(src, dst)

        print(f"Arquivos movidos para: {datasets_path}")
        return datasets_path
