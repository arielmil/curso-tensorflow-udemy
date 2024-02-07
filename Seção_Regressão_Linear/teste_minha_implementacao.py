from minha_implementacao import multiple_gradient_descent_wrapper, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Base com 21 colunas e 21613 linhas aonde cada coluna é um atributo, e cada linha são os valores dos atributos para uma casa.
base = pd.read_csv('house_prices.csv')
print(base.head())
print(f"base.shape: {base.shape}")

#Faz uma matriz com apenas os precos das casas
precos = np.array(base.iloc[:, 2:3])

#Faz uma matriz com todo o resto tirando o preço das casas e a data de construcao.
atributos = np.array(base.iloc[:, 3:21])
print(f"atributos: {atributos}, shape: {atributos.shape}")


epochs = 10000
learning_rate = 0.001
coeficients, errors, previsions = multiple_gradient_descent_wrapper(atributos, precos, epochs, learning_rate)
mae = mean_absolute_error(precos, previsions)

print(f"coeficients: {coeficients},\n\nprevisions[0:10]: {previsions[0:10]},\n\nprecos[0:10]: {precos[0:10]}")
print(f"mae: {mae}")
