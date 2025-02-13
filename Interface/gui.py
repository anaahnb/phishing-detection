import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import PySimpleGUI as sg
import joblib
import pandas as pd
from MachineLearning.model_trainer import PhishingModelTrainer
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

class PhishingDetectorGUI:
    def __init__(self, model_path="Evaluation/phishing_model.pkl"):
        self.model_path = model_path
        self.model = self.load_model()
        self.window = self.create_window()
    
    def load_model(self):
        """Carrega o modelo treinado."""
        return joblib.load(self.model_path)
    
    def create_window(self):
        """Cria a interface gráfica."""
        sg.theme("LightBlue2")
        layout = [
            [sg.Text("Insira a URL para análise:")],
            [sg.InputText(key="URL")],
            [sg.Button("Verificar"), sg.Button("Sair")],
            [sg.Text("", size=(40, 1), key="RESULTADO", text_color="black", justification="center")]
        ]
        return sg.Window("Detecção de Phishing", layout)
    
    def extract_features(self, url):
        """Extrai as features da URL fornecida."""
        df_temp = pd.DataFrame([{"url": url}])
        extractor = UrlFeatureExtractor(df_temp)
        return extractor.extract_all()
    
    def predict(self, features):
        """Realiza a predição da URL."""
        return self.model.predict(features)
    
    def run(self):
        """Executa o loop da interface gráfica."""
        while True:
            event, values = self.window.read()
            if event in (sg.WINDOW_CLOSED, "Sair"):
                break
            if event == "Verificar":
                url = values["URL"]
                if url:
                    df_features = self.extract_features(url)
                    prediction = self.predict(df_features)
                    resultado = "Phishing" if prediction[0] == 1 else "Legítima"
                    color = "red" if prediction[0] == 1 else "green"
                    self.window["RESULTADO"].update(f"Classificação: {resultado}", text_color=color)
        
        self.window.close()

if __name__ == "__main__":
    app = PhishingDetectorGUI()
    app.run()
