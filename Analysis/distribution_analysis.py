import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from DataLoading.data_loader import DataLoader
from Settings.keys import ParamsKeys

data_loader = DataLoader()
df = data_loader.load_data()

df_numeric = df.select_dtypes(include=[np.number])

output_dir = "Figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

print(f'Gráficos salvos em {output_dir}')
