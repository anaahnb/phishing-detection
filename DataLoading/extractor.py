import os
import zipfile

class DataExtractor:
    @staticmethod
    def extract(zip_path: str, extract_to: str) -> str:
        """Extrai um arquivo ZIP para um diretório específico e retorna o caminho extraído."""

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Arquivo extraído para: {extract_to}")
        return extract_to
