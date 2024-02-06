import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Proximo passo: Fazer a descida de gradiente poder ter qualquer número de variaveis independentes.

idades = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
precos = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

#Adiciona uma coluna com 1's no final de idades para representar o coeficiente de b em idades_escaladas:
#axis=1 serve para adicionar a coluna no eixo das colunas.
idades = np.append(idades, np.ones((idades.shape[0], 1)), axis=1)

def mean_squared_error(y, y_pred):
    return ((y - y_pred)**2).mean()

def simple_gradient_descent(x, y, epochs, learning_rate, initial_a, initial_b):
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

#Função que faz a descida de gradiente para um número qualquer de variaveis independentes.
#x contém todos os valores das variaveis independentes.
#y contém o valor da váriavel dependente.
def multiple_gradient_descent(x, y, epochs, learning_rate, coeficients):
    errors = np.array([])
    for _ in range(epochs):
        #Faz y = a1*x1 + a2*x2 + ... + a(n-1)x(n-1) + an
        y_pred = np.dot(x, coeficients)

        #Calcula o erro médio quadrático.
        error = mean_squared_error(y, y_pred)
        errors = np.append(errors, error)

        #Calcula o gradiente para cada coeficiente
        #o .T é para fazer a transposta da matriz x.
        #O .dot é para fazer o produto escalar entre x e y - y_pred.
        #como x contém o valor 1 em todas as linhas para a última coluna, o produto escalar entre essa coluna e y - y_pred é igual a: 
        # -2 * (y - (a1*x1 + a2*x2 + ... + a(n-1)x(n-1) + an)) (Como na função de descida de gradiente simples)
        gradients = -2 * (x.T.dot(y - y_pred)).mean()

        #Atualiza os coeficientes.
        coeficients = coeficients - learning_rate * gradients

    return coeficients, errors


print("Descida de gradiente simples:")
epochs = 10000

#Escala a idade e preço para diminuir a diferença de escala entre os dados e permitir que o gradient_descent seja mais eficiente.
scalar_idade = StandardScaler()
idades_escaladas = scalar_idade.fit_transform(idades)
scalar_preco = StandardScaler()
precos_escalados = scalar_preco.fit_transform(precos)

#Gera numeros aleatorios para inicializar a e b.
np.random.seed(0)
rand = np.random.rand(2)

'''a, b, errors = simple_gradient_descent(idades_escaladas, precos_escalados, epochs, 0.001, rand[0], rand[1])

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
plt.show()'''

#retira a coluna com 1's no final de idades e idades_escaladas:
idades = idades[:, 0]
idades_escaladas = idades_escaladas[:, 0]

#Descida de gradiente para um número qualquer de variaveis independentes:
print("Descida de gradiente para um número qualquer de variaveis independentes:")
coeficients, errors = multiple_gradient_descent(idades_escaladas, precos_escalados, epochs, 0.001, np.random.rand(2))

a = coeficients[0]
b = coeficients[1]

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