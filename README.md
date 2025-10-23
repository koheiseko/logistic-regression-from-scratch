# Regressão Logística do Zero

Este projeto é uma implementação da Regressão Logística para classificação binária, desenvolvida do zero (from scratch) utilizando apenas a biblioteca NumPy para as operações matemáticas e de álgebra linear.

O objetivo é construir um modelo de classificação compreendendo todos os seus componentes internos, como a função sigmóide, a função de custo (Log loss) e o algoritmo de otimização (Gradiente Descendente).

O modelo foi validado em dois conjuntos de dados distintos:

Dados Sintéticos: Um conjunto de dados 2D gerado artificialmente para visualização e validação inicial.

Dataset Titanic (Kaggle): O clássico conjunto de dados "Titanic - Machine Learning from Disaster" do Kaggle, usado para testar o modelo em um cenário de dados reais e mais complexos.

---

## Funcionalidades

- Implementação pura em NumPy, sem uso de bibliotecas de alto nível como Scikit-learn para o modelo principal.
- Otimização dos pesos (coeficientes) através do algoritmo de Gradiente Descendente (Batch).
- Função de custo de Log Loss.
- Métodos fit() para treinamento e predict() para realizar previsões.

---

## Tecnologias Utilizadas

- Python 3.13
- NumPy: Para todos os cálculos matriciais, vetorização e o núcleo do algoritmo.
- Pandas: Para carregamento e pré-processamento do dataset Titanic.
- Scikit-learn: Utilizado apenas para gerar os dados sintéticos (make_classification) e para métricas de avaliação (como accuracy_score, confusion_matrix).
- Matplotlib / Seaborn: Para visualização dos dados sintéticos.
