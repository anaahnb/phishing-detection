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
from Settings.keys import ParamsKeys

warnings.filterwarnings("ignore")

class XGBoostHyperparameterTuning:
    def __init__(self):
        """Define a grade de hiperparâmetros para ajuste e configurações iniciais."""
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

        self.param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1, 1.5, 2.0]
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
        """Pré-processa os dados com divisão consistente de outliers."""
        preprocessor = DataPreprocessor(df)
        train_set, test_set, val_set = preprocessor.clean_and_split()

        if "url" in train_set.columns:
            train_set = train_set.drop(columns=[ParamsKeys.URL])
            test_set = test_set.drop(columns=[ParamsKeys.URL])
            val_set = val_set.drop(columns=[ParamsKeys.URL])

        X_train, y_train = train_set.drop(columns=[ParamsKeys.STATUS]), train_set[ParamsKeys.STATUS]
        X_test, y_test = test_set.drop(columns=[ParamsKeys.STATUS]), test_set[ParamsKeys.STATUS]
        X_val, y_val = val_set.drop(columns=[ParamsKeys.STATUS]), val_set[ParamsKeys.STATUS]

        return X_train, X_test, X_val, y_train, y_test, y_val

    def tune_hyperparameters(self):
        """Executa o Grid Search para encontrar os melhores hiperparâmetros."""
        df = self.load_data()
        X_train, X_test, X_val, y_train, y_test, y_val = self.preprocess_data(df)

        print("\nIniciando o ajuste de hiperparâmetros...")

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
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

        # Avaliação no conjunto de validação
        y_val_pred = self.model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)
        print(f"F1-score no conjunto de validação: {val_f1:.4f}")

        # Avaliação final no conjunto de teste
        y_test_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)

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
