import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor
from Settings.keys import ParamsKeys
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

class PhishingModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.output_dir = "Evaluation"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        data_loader = DataLoader()
        df = data_loader.load_data()
        return df

    def preprocess_data(self, df):
        preprocessor = DataPreprocessor(df)
        return preprocessor.preprocess()

    def train(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia do modelo: {accuracy:.4f}")
        print("Relatório de classificação:\n", classification_report(y_test, y_pred))

        model_path = os.path.join(self.output_dir, "phishing_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"Modelo salvo em {model_path}")

if __name__ == "__main__":
    trainer = PhishingModelTrainer()
    trainer.train()
