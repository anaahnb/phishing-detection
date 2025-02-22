import os
import sys
import joblib
import warnings
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from Settings.keys import ParamsKeys
from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor

warnings.filterwarnings("ignore")

class XGBoostTrainer:
    def __init__(self):
        """Inicializa o classificador XGBoost com os melhores hiperparâmetros encontrados."""
        self.model = xgb.XGBClassifier(
            colsample_bytree=1.0,
            gamma=0.1,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=300,
            reg_alpha=0,
            reg_lambda=1,
            subsample=0.7,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        self.output_dir = "Evaluation"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """Carrega os dados de phishing."""
        data_loader = DataLoader()
        df = data_loader.load_data()
        return df

    def preprocess_data(self, df):
        """Pré-processa os dados e divide em conjuntos de treino, validação e teste."""
        preprocessor = DataPreprocessor(df)
        train_set, test_set, val_set = preprocessor.clean_and_split()

        # Remover a coluna de URL se existir
        if ParamsKeys.URL in train_set.columns:
            train_set = train_set.drop(columns=[ParamsKeys.URL])
            test_set = test_set.drop(columns=[ParamsKeys.URL])
            val_set = val_set.drop(columns=[ParamsKeys.URL])

        # Garantir que apenas as features extraídas sejam usadas
        feature_columns = [col for col in train_set.columns if col != ParamsKeys.STATUS]
        X_train, y_train = train_set[feature_columns], train_set[ParamsKeys.STATUS]
        X_test, y_test = test_set[feature_columns], test_set[ParamsKeys.STATUS]

        return X_train, X_test, y_train, y_test

    def train_model(self):
        """Treina o modelo e avalia seu desempenho no conjunto de teste."""
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)

        print("\n✅ Iniciando o treinamento do modelo XGBoost...")

        # Treinamento do modelo
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n📊 Desempenho do Modelo no Conjunto de Teste:")
        print(f"   🔹 Acurácia: {accuracy:.4f}")
        print(f"   🔹 Precisão: {precision:.4f}")
        print(f"   🔹 Recall: {recall:.4f}")
        print(f"   🔹 F1-score: {f1:.4f}")

        model_path = os.path.join(self.output_dir, "xgboost_trained_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"\n💾 Modelo treinado salvo em: {model_path}")

if __name__ == "__main__":
    trainer = XGBoostTrainer()
    trainer.train_model()
