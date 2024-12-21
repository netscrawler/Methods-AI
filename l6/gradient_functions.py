#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functions import sigmoid, add_zero_feature, pack_params, unpack_params
from sigmoid_gradient import sigmoid_gradient


def gradient_function(
    nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef
):

    # Распаковка параметров Theta1 и Theta2
    Theta1, Theta2 = unpack_params(
        nn_params, input_layer_size, hidden_layer_size, num_labels
    )

    # Количество примеров
    m = X.shape[0]

    # Прямое распространение
    A_1 = X
    Z_2 = np.dot(A_1, Theta1.T)  # Промежуточные активации для второго слоя
    A_2 = sigmoid(Z_2)  # Активируем второй слой
    A_2 = add_zero_feature(A_2)  # Добавляем единичный столбец для нейрона смещения
    Z_3 = np.dot(A_2, Theta2.T)  # Промежуточные активации для третьего слоя
    A_3 = sigmoid(Z_3)  # Активация выходного слоя

    # Вычисление ошибок по нейронам
    DELTA_3 = A_3 - Y  # Ошибка на выходном слое
    DELTA_2 = np.dot(DELTA_3, Theta2[:, 1:]) * sigmoid_gradient(
        Z_2
    )  # Ошибка на скрытом слое

    # Вычисление частных производных
    D1 = np.dot(DELTA_2.T, A_1) / m  # Градиент для Theta1
    D2 = np.dot(DELTA_3.T, A_2) / m  # Градиент для Theta2

    # Добавление регуляризации
    D1[:, 1:] += (lambda_coef / m) * Theta1[:, 1:]  # Регуляризация для Theta1
    D2[:, 1:] += (lambda_coef / m) * Theta2[:, 1:]  # Регуляризация для Theta2

    return pack_params(D1, D2)
