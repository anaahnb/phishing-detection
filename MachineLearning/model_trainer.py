import os
import sys
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

class PhishingModelTrainer:
    def __init__(self):

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=9,
            learning_rate=0.1,
            colsample_bytree=0.7,
            eval_metric="logloss",
            random_state=42
        )
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
        """Treina o modelo XGBoost e avalia seu desempenho."""
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)

        print("\nTreinando o modelo XGBoost...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nDesempenho do Modelo Final XGBoost:")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

        model_path = os.path.join(self.output_dir, "xgboost_phishing_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"\nModelo salvo em: {model_path}")

if __name__ == "__main__":
    trainer = PhishingModelTrainer()
    trainer.train()
