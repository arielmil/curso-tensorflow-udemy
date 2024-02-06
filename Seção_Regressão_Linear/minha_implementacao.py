import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Faz uma matriz 10x1 (dez linhas e uma coluna) com os valores de 18 a 63.
idades = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])

#Faz uma matriz 10x1 (dez linhas e uma coluna) com os valores de 871 a 1900.
precos = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

def mean_squared_error(y, y_pred):
    return ((y - y_pred)**2).mean()

def simple_gradient_descent(x, y, epochs, learning_rate, a, b, debug=False):
    errors = np.array([])

    for i in range(epochs):
        y_pred = a * x + b
        error = mean_squared_error(y, y_pred)
        
        if debug:
            print("Epoch: ", i, "Error: ", error, "a: ", a, "b: ", b)
        
        errors = np.append(errors, error)

        gradient_a = -2 * (x * (y - (b + a * x) ) ).mean()
        gradient_b = -2 * (y - (b + a * x)).mean()
    
        a = a - learning_rate * gradient_a
        b = b - learning_rate * gradient_b
    return a, b, errors

#Função que faz a descida de gradiente para um número qualquer de variaveis independentes.
#x contém todos os valores das variaveis independentes.
#y contém o valor da váriavel dependente.
def multiple_gradient_descent(x, y, epochs, learning_rate, coeficients, debug=False):
    errors = np.array([])
    for i in range(epochs):
        #Faz y = a1*x1 + a2*x2 + ... + a(n-1)x(n-1) + an
        y_pred = np.dot(x, coeficients)

        #Calcula o erro médio quadrático.
        error = mean_squared_error(y, y_pred)
        errors = np.append(errors, error)

        #Calcula o gradiente para cada coeficiente
        #o .T é para fazer a transposta da matriz x. (Transformando uma matriz MxN em uma matriz NxM)
        #O .dot é para fazer o produto escalar entre x e y - y_pred.
        #como x contém o valor 1 em todas as linhas para a última coluna, o produto escalar entre essa coluna e y - y_pred é igual a: 
        # -2 * (y - (a1*x1 + a2*x2 + ... + a(n-1)x(n-1) + an)) (Como na função de descida de gradiente simples)
        m = x.shape[0]
        gradients = -2/m * (x.T.dot(y - y_pred))

        #Atualiza os coeficientes.
        coeficients = coeficients - learning_rate * gradients

        if debug:
            print("Epoch: ", i, "Error: ", error, "a: ", a, "b: ", b)
            print("y_pred: ", y_pred)
    
    return coeficients, errors


print("Descida de gradiente simples:")
epochs = 10000
learning_rate = 0.001

#Escala a idade e preço para diminuir a diferença de escala entre os dados e permitir que o gradient_descent seja mais eficiente.

scalar_idade = StandardScaler()
idades_escaladas = scalar_idade.fit_transform(idades)
scalar_preco = StandardScaler()
precos_escalados = scalar_preco.fit_transform(precos)

#Gera numeros aleatorios para inicializar a e b.
#np.random.seed()
rand = np.random.rand(2)

a, b, errors = simple_gradient_descent(idades_escaladas, precos_escalados, epochs, learning_rate, rand[0], rand[1])

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

plt.scatter(idades, precos, color='blue')
plt.title('Idade x Preço x previsao')

#Grafico com a reta de regressão linear:

plt.plot(idades, previsoes, color='red')
plt.show()

#Descida de gradiente para um número qualquer de variaveis independentes:
print("Descida de gradiente para um número qualquer de variaveis independentes:")

#Para cada linha na matriz 10x1 de idades, adiciona uma coluna com o valor 1, fazendo ela se tornar uma matriz 10x2 com [[idadei, 1]] para todo i.
idades_escaladas_com_1 = np.c_[idades_escaladas, np.ones(idades_escaladas.shape[0])]

#Faz uma matriz 2x1 (duas linhas e uma coluna) com números aleatórios.
rand = np.random.rand(2)
coeficients = np.array([[rand[0]], [rand[1]]])

coeficients, errors = multiple_gradient_descent(idades_escaladas_com_1, precos_escalados, epochs, learning_rate, coeficients)

a = coeficients[0]
b = coeficients[1]

#Desescala os dados para que possam ser comparados com os dados brutos.
previsoes = scalar_preco.inverse_transform(np.dot(idades_escaladas_com_1, coeficients))

print("a: ", a)
print("b: ", b)
print("errors: ", errors)

print(precos)
print(previsoes)

#Grafico com os erros:
plt.plot(range(epochs), errors)
plt.title('Erros')
plt.show()

#Grafico com os dados brutos:

plt.scatter(idades, precos, color='blue')
plt.title('Idade x Preço x previsao')

#Grafico com a reta de regressão linear:

plt.plot(idades, previsoes, color='red')
plt.show()