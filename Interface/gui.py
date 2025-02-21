import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess
import PySimpleGUI as sg

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

from DataEngineering.preprocessing import DataPreprocessor

MODEL_PATH = "Evaluation/xgboost_best_model.pkl"
TUNING_SCRIPT = "MachineLearning/hyperparameter_tuning.py"

def check_and_generate_model():
    """Verifica se o modelo treinado existe. Se não existir, executa o script de ajuste de hiperparâmetros."""
    if not os.path.exists(MODEL_PATH):
        print("Modelo não encontrado. Executando ajuste de hiperparâmetros...")
        result = subprocess.run(["python", TUNING_SCRIPT], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Erro ao gerar o modelo. Verifique o script de ajuste de hiperparâmetros.")

def load_model():
    """Carrega o modelo treinado."""
    return joblib.load(MODEL_PATH)

def classify_url(url, model):
    """Classifica a URL fornecida usando o modelo carregado."""

    # Criar um DataFrame com a URL fornecida
    df = pd.DataFrame([url], columns=["url"])

    # Passar o DataFrame para o extrator de features
    extractor = UrlFeatureExtractor(df)
    features_df = extractor.extract_all()

    # Carregar os nomes das features esperadas
    expected_features = model.get_booster().feature_names
    print("Features de treino:", model.get_booster().feature_names)

    # Garantir que todas as features esperadas estão presentes (mesmo que com valor 0)
    features_df = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)

    # Garantir que as features estão na ordem correta
    features_df = features_df[expected_features]
    print(features_df.head())  # Ver se as features estão corretas
    print(features_df.describe())
    # Fazer a predição
    prediction = model.predict(features_df.values)

    return "Phishing" if prediction == 1 else "Legítima"

def main():
    check_and_generate_model()
    model = load_model()

    sg.theme("DarkBlue")
    layout = [
        [sg.Text("Insira a URL para verificação:")],
        [sg.InputText(key="-URL-")],
        [sg.Button("Verificar"), sg.Button("Sair")],
        [sg.Text("", size=(30,1), key="-OUTPUT-")]
    ]

    window = sg.Window("Detecção de Phishing", layout)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Sair"):
            break
        if event == "Verificar":
            url = values["-URL-"]
            if url:
                resultado = classify_url(url, model)
                window["-OUTPUT-"].update(f"Resultado: {resultado}")
            else:
                window["-OUTPUT-"].update("Por favor, insira uma URL.")

    window.close()

if __name__ == "__main__":
    main()
