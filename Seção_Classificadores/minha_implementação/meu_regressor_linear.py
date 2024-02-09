import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def mean_squared_error(y, y_pred):
    return ((y - y_pred)**2).mean()

def mean_absolute_error(y, y_pred):
    return np.abs(y - y_pred).mean()

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
def multiple_gradient_descent(x, y, epochs, learning_rate, coeficients, batch_size = False, debug=False):
    errors = np.array([])
    
    '''
    if batch_size:

        #gera sbatch números aleatórios para que a descida de gradiente seja feita em batchs.
        random_indexes = np.random.choice(x.shape[0], batch_size, replace=False)
        
        #Para cada valor i em random_batches, faz X = x[i] e Y = y[i]
        new_x = [x[i] for i in random_indexes]
        new_y = [y[i] for i in random_indexes]
    '''

        
    
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
            for i in range(coeficients.shape[0]):
                print(f"Epoch: {i} Error: {error} Coeficients: {coeficients}")

    return coeficients, errors


def multiple_gradient_descent_wrapper(X, y, epochs, learning_rate, debug=False):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    #Escala X e y para ficaram em ordem de magnitudes semelhantes
    scaled_X = scaler_X.fit_transform(X)
    scaled_y = scaler_y.fit_transform(y)

    #Para cada linha na matriz MxN, adiciona uma coluna com o valor 1, fazendo ela se tornar uma matriz MxN+1.
    scaled_X_with_1 = np.c_[scaled_X, np.ones(scaled_X.shape[0])]

    #Gera n + 1 valores aleatorios para os coeficientes da funcao y = a1*x1 + a2*x2 + ... + an*xn + b
    rand = np.random.rand(scaled_X_with_1.shape[1])
    coefs = np.array([rand]).T

    coeficients, errors = multiple_gradient_descent(scaled_X_with_1, scaled_y, epochs, learning_rate, coefs, debug)

    #Desescala y_previsto atravez do escalador de y, e aplica a formula y = X o coeficientes, que é igual a: y = a1*x1 + a2*x2 + ... + an*xn + b
    previsions = scaler_y.inverse_transform(np.dot(scaled_X_with_1, coeficients))

    if debug:
        for i in range(coeficients.shape[0]):
            print(f"coeficiente {i}: {coeficients[i]}")

            print(f"y: {y}")
            print(f"previsions: {previsions}")
            
    #Grafico com os erros:
    plt.plot(range(epochs), errors)
    plt.xlabel('Época')
    plt.ylabel('Erro')
    plt.title('Erros x epochs')
    plt.show()

    return coeficients, errors, previsions, scaler_y
		
		
		

'''print("Descida de gradiente simples:")
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
plt.show()'''
