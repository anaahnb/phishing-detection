import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor


class ModelEvaluator:
    def __init__(self, model_path: str, data_loader: DataLoader, preprocessor: DataPreprocessor):
        """
        Inicializa a classe para avaliação do modelo.

        :param model_path: Caminho do modelo treinado (arquivo .pkl)
        :param data_loader: Instância do DataLoader para carregar os dados
        :param preprocessor: Instância do DataPreprocessor para pré-processamento
        """
        self.model_path = model_path
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = self.load_model()

        # Carregar e processar os dados
        self.df = self.data_loader.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.preprocess()

    def load_model(self):
        """Carrega o modelo treinado a partir do arquivo."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        return joblib.load(self.model_path)

    def evaluate(self):
        """Executa a avaliação do modelo com matriz de confusão, curva ROC e relatório de métricas."""
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]  # Probabilidade da classe phishing (1)

        self.plot_confusion_matrix(y_pred)
        self.plot_roc_curve(y_proba)
        self.print_classification_report(y_pred)

    def plot_confusion_matrix(self, y_pred):
        """Plota a matriz de confusão."""
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legítimo", "Phishing"], yticklabels=["Legítimo", "Phishing"])
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.show()

    def plot_roc_curve(self, y_proba):
        """Plota a curva ROC e calcula a AUC."""
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("Falso Positivo Rate")
        plt.ylabel("Verdadeiro Positivo Rate")
        plt.title("Curva ROC")
        plt.legend()
        plt.show()

    def print_classification_report(self, y_pred):
        """Exibe o relatório de classificação com métricas de desempenho."""
        print("Relatório de Classificação:\n", classification_report(self.y_test, y_pred))


# Exemplo de uso:
if __name__ == "__main__":
    model_path = os.path.join("Evaluation", "xgboost_best_model.pkl")

    data_loader = DataLoader()
    preprocessor = DataPreprocessor(data_loader.load_data())

    evaluator = ModelEvaluator(model_path, data_loader, preprocessor)
    evaluator.evaluate()
