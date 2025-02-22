import os
import sys
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from DataLoading.data_loader import DataLoader
from DataEngineering.preprocessing import DataPreprocessor
from Settings.keys import ParamsKeys
from FeatureEngineering.url_feature_extractor import UrlFeatureExtractor

class PhishingModelComparison:
  def __init__(self):
    self.models = {
      "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
      "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
      "SVM": LinearSVC(random_state=42),
      "Regressão Logística": LogisticRegression(max_iter=500, random_state=42)
    }
    self.results = {}

  def load_data(self):
    data_loader = DataLoader()
    df = data_loader.load_data()
    return df

  def preprocess_data(self, df):
    preprocessor = DataPreprocessor(df)
    return preprocessor.clean_and_split()

  def train_and_evaluate(self):
    df = self.load_data()
    X_train, X_test, y_train, y_test = self.preprocess_data(df)

    for name, model in self.models.items():
      print(f"\nTreinando {name}...")
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)

      self.results[name] = {
          "Acurácia": accuracy,
          "Precisão": precision,
          "Recall": recall,
          "F1-score": f1
      }

      print(f"Resultados para {name}:")
      print(classification_report(y_test, y_pred))

    self.save_results()

  def save_results(self):
    results_df = pd.DataFrame.from_dict(self.results, orient="index")
    results_path = os.path.join("Evaluation", "model_comparison_results.csv")

    if not os.path.exists("Evaluation"):
        os.makedirs("Evaluation")

    results_df.to_csv(results_path, index=True)
    print(f"\nResultados salvos em: {results_path}")

if __name__ == "__main__":
  comparison = PhishingModelComparison()
  comparison.train_and_evaluate()
