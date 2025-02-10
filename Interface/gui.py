import PySimpleGUI as sg
import joblib
import pandas as pd
from MachineLearning.model_trainer import PhishingModelTrainer
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

# Carregar o modelo treinado
model_path = "MachineLearning/Evaluation/phishing_model.pkl"
model = joblib.load(model_path)

sg.theme("LightBlue2")
layout = [
    [sg.Text("Insira a URL para análise:")],
    [sg.InputText(key="URL")],
    [sg.Button("Verificar"), sg.Button("Sair")],
    [sg.Text("", size=(40, 1), key="RESULTADO", text_color="black", justification="center")]
]

# Criar a janela
window = sg.Window("Detecção de Phishing", layout)

# Loop de eventos
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Sair":
        break

    if event == "Verificar":
        url = values["URL"]

        if url:
            # Criar um DataFrame temporário para extrair features da URL
            df_temp = pd.DataFrame([{"url": url}])
            extractor = UrlFeatureExtractor(df_temp)
            df_features = extractor.extract_all()

            # Fazer a predição
            prediction = model.predict(df_features)

            # Exibir o resultado
            resultado = "Phishing" if prediction[0] == 1 else "Legítima"
            window["RESULTADO"].update(f"Classificação: {resultado}", text_color="red" if prediction[0] == 1 else "green")

# Fechar a janela
window.close()
