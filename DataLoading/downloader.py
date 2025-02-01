import kagglehub
from Settings.keys import ParamsKeys

class DataDownloader:
    @staticmethod
    def download() -> str:
        """Baixa a versão mais recente do dataset do Kaggle e retorna o caminho dos arquivos."""

        path = kagglehub.dataset_download(ParamsKeys.DATASET_NAME)
        print(f"Arquivos baixados para: {path}")
        return path
