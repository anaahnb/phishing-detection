import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from data.data_loader import load_data

df = load_data()

df_numeric = df.select_dtypes(include=[np.number])

corr_matrix = df_numeric.corr()

output_dir = "analysis/results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis')
plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight")
print('A matriz foi salva em analysis/results')

