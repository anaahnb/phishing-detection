import os
import sys
import joblib
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor
from Settings.keys import ParamsKeys
warnings.filterwarnings("ignore")

class ModelValidator:
    def __init__(self, model_path="Evaluation/xgboost_trained_model.pkl"):
        """Carrega o modelo XGBoost treinado para validação."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Execute o treinamento primeiro.")

        self.model = joblib.load(model_path)
        self.model_path = model_path
        print(f"Modelo carregado com sucesso de {model_path}.")

    def load_data(self):
        """Carrega os dados brutos."""
        data_loader = DataLoader()
        df = data_loader.load_data()
        return df

    def preprocess_data(self, df):
        """Processa os dados e obtém o conjunto de validação."""
        preprocessor = DataPreprocessor(df)
        train_set, test_set, val_set = preprocessor.clean_and_split()

        if "url" in train_set.columns:
            train_set = train_set.drop(columns=[ParamsKeys.URL])
            test_set = test_set.drop(columns=[ParamsKeys.URL])
            val_set = val_set.drop(columns=[ParamsKeys.URL])

        # Separando features (X) e rótulos (y) do conjunto de validação
        X_val, y_val = val_set.drop(columns=["status"]), val_set["status"]

        return X_val, y_val

    def validate_model(self):
        """Valida o modelo no conjunto de validação e exibe as métricas de desempenho."""
        df = self.load_data()
        X_val, y_val = self.preprocess_data(df)

        print("\nRealizando validação do modelo...")

        # Realizando predição no conjunto de validação
        y_pred = self.model.predict(X_val)

        # Cálculo das métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        print("\nDesempenho do Modelo no Conjunto de Validação:")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Matriz de confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legítimo", "Phishing"], yticklabels=["Legítimo", "Phishing"])
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão - Validação")
        plt.show()

if __name__ == "__main__":
    validator = ModelValidator()
    validator.validate_model()
