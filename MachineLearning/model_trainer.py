import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from DataLoading import DataLoader
from Settings.keys import ParamsKeys

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
        X = df.drop(columns=ParamsKeys.STATUS)
        y = df["status"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

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
