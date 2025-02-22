import os
import sys
import joblib
import pandas as pd
import warnings
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

warnings.filterwarnings("ignore")

class PhishingPredictor:
    def __init__(self, model_path="Evaluation/xgboost_trained_model.pkl"):
        """Inicializa o preditor carregando o modelo treinado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Execute o treinamento primeiro.")

        self.model = joblib.load(model_path)
        print(f"Modelo carregado com sucesso de {model_path}.")

    def extract_features(self, url):
        """Extrai as features da URL fornecida."""
        df = pd.DataFrame([{"url": url}])  # Criamos um DataFrame com a URL
        feature_extractor = UrlFeatureExtractor(df)
        extracted_features = feature_extractor.extract_all()

        # Remove colunas desnecessárias, mantendo apenas as features
        feature_cols = [col for col in extracted_features.columns if col != "url"]
        return extracted_features[feature_cols]


    def predict(self, url):
        """Realiza a predição da URL informada e retorna o resultado."""
        features = self.extract_features(url)
        prediction = self.model.predict(features)[0]

        result = "PHISHING" if prediction == 1 else "LEGÍTIMA"
        print(f"\nResultado da predição para a URL:\n{url} → {result}")
        return result

if __name__ == "__main__":
    predictor = PhishingPredictor()

    # Teste com uma URL
    test_url = input("Digite a URL para verificação: ")
    predictor.predict(test_url)
