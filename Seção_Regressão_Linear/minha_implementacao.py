import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

idades = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
precos = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

def mean_squared_error(y, y_pred):
    return ((y - y_pred)**2).mean()

def gradient_descent(x, y, epochs, learning_rate, initial_a, initial_b):
    a = initial_a
    b = initial_b
    errors = np.array([])

    for _ in range(epochs):
        y_pred = a * x + b
        error = mean_squared_error(y, y_pred)
        #print("Epoch: ", i, "Error: ", error, "a: ", a, "b: ", b)
        
        errors = np.append(errors, error)

        gradient_a = -2 * (x * (y - (b + a * x) ) ).mean()
        gradient_b = -2 * (y - (b + a * x)).mean()
    
        a = a - learning_rate * gradient_a
        b = b - learning_rate * gradient_b
    return a, b, errors

epochs = 10000

#Escala a idade e preço para diminuir a diferença de escala entre os dados e permitir que o gradient_descent seja mais eficiente.
scalar_idade = StandardScaler()
idades_escaladas = scalar_idade.fit_transform(idades)
scalar_preco = StandardScaler()
precos_escalados = scalar_preco.fit_transform(precos)

#Gera numeros aleatorios para inicializar a e b.
np.random.seed(0)
rand = np.random.rand(2)

a, b, errors = gradient_descent(idades_escaladas, precos_escalados, epochs, 0.001, rand[0], rand[1])

#Desescala os dados para que possam ser comparados com os dados brutos.
previsoes = scalar_preco.inverse_transform(a*idades_escaladas + b)

print("a: ", a)
print("b: ", b)
print("errors: ", errors)

print(precos)
print(previsoes)

#Grafico com os erros:
print(epochs, errors.shape[0])
plt.plot(range(epochs), errors)
plt.title('Erros')
plt.show()

#Grafico com os dados brutos:

plt.scatter(idades, precos, color='b')
plt.title('Idade x Preço x previsao')

#Grafico com a reta de regressão linear:

plt.plot(idades, previsoes, color='r')
plt.show()
