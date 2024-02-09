import pandas as pd
from meu_classificador import multiple_logistic_regression_wrapper
base = pd.read_csv('census.csv')

X = base.iloc[:, 0:14].values
y = base.iloc[:, 14].values

epochs = 2000
learning_rate = 0.01
coeficients, errors = multiple_logistic_regression_wrapper(X, y, epochs, learning_rate)

print(f"coeficients: {coeficients}")
print(f"errors.mean(): {errors.mean()}")