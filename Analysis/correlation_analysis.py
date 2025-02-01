import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from DataLoading.data_loader import DataLoader

data_loader = DataLoader()
df = data_loader.load_data()

df_numeric = df.select_dtypes(include=[np.number])

corr_matrix = df_numeric.corr()

output_dir = "Figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis')
plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches="tight")
print(f'A matriz foi salva em {output_dir}')

