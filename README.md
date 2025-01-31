# Phishing Detection

## Instalação
Requisitos: [Python 3](https://www.python.org/download/releases/3.0/)

Crie um ambiente virtual e instale as bibliotecas necessárias.
```bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate

pip install -r requirements.txt
```

## Execução

Estando dentro do ambiente virtual, utilize os comandos abaixo

### Para executar o processo completo de pré-processamento
Esse script irá carregar os dados, limpá-los, extrair características das URLs (features) e salvar os dados processados.
```bash
python phishing_detection.py
```

### Para visualizar as análises realizadas:

Esse script apresenta uma **análise de correlação entre as variáveis numéricas** para entender como elas se relacionam. Ao executá-lo, um gráfico será gerado dentro da pasta ```analysis/results```.
```bash
python -m analysis.correlation_analysis
```

Esse script apresenta a **análise da distribuição das variáveis** do conjunto de dados. Ele gera gráficos sobre a distribuição da variável alvo e das variáveis numéricas, incluindo histogramas e boxplots. Ao executá-lo, gráficos serão gerados dentro da pasta ```analysis/results```.

```bash
python -m analysis.distribution_analysis
```