from Seção_Regressão_Linear.minha_implementação.meu_regressor_linear import multiple_gradient_descent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Função que calcula o erro probabilístico 
#retorna valores muito altos a medida que a previsão se distancia do valor real e valores muito baixos a medida que a previsão se aproxima do valor real
def log_loss(y, y_pred):
    return -( np.mean( y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred) ) )

#Função que calcula a função sigmoide em cima de um valor x que será a equação linear que queremos transformar em uma função probabilística.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def multiple_logistic_regression(x, y, epochs, learning_rate, coeficients, debug=False):
    errors = np.array([])
    
    for i in range(epochs):
        x_with_ones = np.c_[np.ones(x.shape[0]), x]
        coeficients = multiple_gradient_descent(x_with_ones, y, coeficients, learning_rate)
        reta_linear = np.dot(x_with_ones, coeficients)

        #Deslineariza a reta transformando ela numa função de sigmoid (retiorna valores entre 0 e 1, que é o que queremos para uma função de probabilidade)
        y_pred = sigmoid(reta_linear)
        error = log_loss(y, y_pred)
        errors = np.append(errors, error)
        
        #Gradiente da função log_loss em relação aos coeficientes (ver depois por que as derivadas parciais de log_loss em relacao aos coefs deu isso.)
        #Fazer uma outra versão que gere a função gradiente automaticamente.
        gradient = np.dot(x.T, (y_pred - y)) / y.size
        
        coeficients -= learning_rate * gradient
        
        if debug:
            print(f"Epoch: {i}, Error: {error}, Coeficients: {coeficients}")

    return coeficients, errors

def multiple_logistic_regression_wrapper(x, y, epochs, learning_rate, debug=False):
    #Adiciona uma coluna de 1s para o coeficiente de b
    coeficients = np.random.rand(x.shape[1])

    #Para cada coluna de atributos, que é um tipo de dado categórico, transforma em um número inteiro.
    #Isso é necessário para que a função sigmoid funcione.
    for i in range(x.shape[1]):
        if (type(x[0, i]) == str):
            label_encoder_x = LabelEncoder()
            x[:, i] = label_encoder_x.fit_transform(x[:, i])
    
    if (type(y[0]) == str):
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)

    #Escalona os atributos de entrada para que a função sigmoid funcione melhor.
    #Não precisa escalonar o y pois ele é um valor inteiro que representa a sua classe, logo não faz sentido escalonar.
    scaler_x = StandardScaler()
    scaled_x = scaler_x.fit_transform(x)

    coeficients, errors = multiple_logistic_regression(scaled_x, y, epochs, learning_rate, coeficients, debug)

    #Retransforma o atributo em y em um atributo categorico.
    if (type(y[0]) == str):
        y = label_encoder_y.inverse_transform(y)

    #Plota um gráfico do erro em relação as épocas.
    plt.plot(errors)
    plt.title('Erro por época')
    plt.xlabel('Época')
    plt.ylabel('Erro')
    plt.show()
    
    return coeficients, errors
