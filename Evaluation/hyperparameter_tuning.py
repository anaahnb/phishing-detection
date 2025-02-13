import os
import sys
import joblib
import warnings

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor

warnings.filterwarnings("ignore")

class XGBoostHyperparameterTuning:
    def __init__(self):
        """Define a grade de hiperparâmetros para ajuste e configurações iniciais."""
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

        self.param_grid = {
            "n_estimators": [100, 200, 300],  # Número de árvores
            "max_depth": [3, 6, 9],           # Profundidade das árvores
            "learning_rate": [0.01, 0.1, 0.3], # Taxa de aprendizado
            "subsample": [0.7, 1.0],          # Amostragem de dados para cada árvore
            "colsample_bytree": [0.7, 1.0]    # Amostragem de features por árvore
        }

        self.output_dir = "Evaluation"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """Carrega os dados de phishing."""
        data_loader = DataLoader()
        df = data_loader.load_data()
        return df

    def preprocess_data(self, df):
        """Pré-processa os dados."""
        preprocessor = DataPreprocessor(df)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        return X_train, X_test, y_train, y_test

    def tune_hyperparameters(self):
        """Executa o Grid Search para encontrar os melhores hiperparâmetros."""
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)

        print("\nIniciando o ajuste de hiperparâmetros...")

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring="f1",  # Otimizamos para o F1-score (equilíbrio entre precisão e recall)
            cv=3,  # Validação cruzada com 3 divisões
            n_jobs=-1,  # Usa todos os núcleos disponíveis
            verbose=2
        )

        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        print("\nMelhores hiperparâmetros encontrados:", best_params)

        # Treinar modelo com os melhores hiperparâmetros
        self.model = XGBClassifier(
            **best_params, use_label_encoder=False, eval_metric="logloss", random_state=42
        )
        self.model.fit(X_train, y_train)

        # Avaliar modelo final
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nDesempenho do Modelo Otimizado:")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Salvar modelo treinado
        model_path = os.path.join(self.output_dir, "xgboost_best_model.pkl")
        joblib.dump(self.model, model_path)
        print(f"\nModelo otimizado salvo em: {model_path}")

if __name__ == "__main__":
    tuner = XGBoostHyperparameterTuning()
    tuner.tune_hyperparameters()
