#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from cost_function import cost_function
from sigmoid_gradient import sigmoid_gradient
from gradient_function import gradient_function
from functions import sigmoid, add_zero_feature, decode_y, rand_initialize_weights, pack_params, unpack_params

if __name__ == '__main__':
    # Задание 1. Загрузить обучающую выборку и весовые коэффициенты
    data = loadmat('training_set.mat')
    X = data['X']  # 5000 примеров по 400 признаков (изображение 20x20)
    y = data['y']  # Вектор меток

    weights = loadmat('weights.mat')
    Theta1 = weights['Theta1']  # Матрица весов для первого слоя (25x401)
    Theta2 = weights['Theta2']  # Матрица весов для второго слоя (10x26)

    # Задание 2. Определить параметры нейронной сети
    input_layer_size = 400  # 20x20 изображение
    hidden_layer_size = 25   # 25 нейронов в скрытом слое
    num_labels = 10          # 10 классов (цифры от 0 до 9)
    m = X.shape[0]           # Количество примеров (5000)

    # добавление единичного столбца (bias) к X
    X = add_zero_feature(X)  # Теперь X имеет размерность (5000, 401)

    # декодирование вектора меток y в формат one-hot
    Y = decode_y(y)  # Y имеет размерность (5000, 10)

    # Упаковка матриц Theta1 и Theta2 в один вектор
    nn_params = pack_params(Theta1, Theta2)

    # Проверка функции стоимости для lambda = 0 и lambda = 1
    for lambda_coef in [0, 1]:
        cost = cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef)
        print(f'Функция стоимости для lambda {lambda_coef} = {cost}')

    # Проверка производной функции sigmoid
    gradient = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Производная функции sigmoid в точках -1, -0.5, 0, 0.5, 1:')
    print(gradient)

    # Случайная инициализация весов
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = pack_params(initial_Theta1, initial_Theta2)

    # Обучение нейронной сети
    lambda_coef = 1
    result = minimize(cost_function, initial_nn_params, method='L-BFGS-B', jac=gradient_function, options={'maxiter': 100},
                      args=(input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef))
    
    # Извлечение оптимизированных параметров Theta1 и Theta2
    Theta1, Theta2 = unpack_params(result.x, input_layer_size, hidden_layer_size, num_labels)

    # Вычисление активаций для всех примеров из обучающей выборки
    h1 = sigmoid(np.dot(X, Theta1.T))  # Размерность h1: (5000, 25)
    h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))  # Размерность h2: (5000, 10)
    
    # Предсказание классов для каждого примера
    y_pred = np.argmax(h2, axis=1) + 1  # +1, так как классы начинаются с 1

    # Вычисление точности модели
    accuracy = np.mean(y_pred == y.ravel()) * 100
    print(f'Точность нейронной сети на обучающей выборке: {accuracy:.2f}%')

