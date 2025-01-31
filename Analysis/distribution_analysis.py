import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from data.data_loader import load_data
from settings.keys import ParamsKeys

df = load_data()
df_numeric = df.select_dtypes(include=[np.number])

output_dir = "analysis/results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(6, 4))
sns.countplot(x=ParamsKeys.STATUS, data=df, palette='Set2')
plt.title('Distribuição da Variável Alvo')
plt.savefig(f"{output_dir}/target_distribution.png", dpi=300, bbox_inches="tight")

num_cols = df_numeric.columns[:6]
df_numeric[num_cols].hist(bins=30, figsize=(12, 10), color='steelblue', edgecolor='black')
plt.suptitle('Distribuição das Variáveis Numéricas')
plt.savefig(f"{output_dir}/variable_distribution.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_numeric[num_cols], orient='h', palette='Set2')
plt.title('Boxplots das Variáveis Numéricas')
plt.savefig(f"{output_dir}/boxplots.png", dpi=300, bbox_inches="tight")

print('Gráficos salvos em analysis/results')
