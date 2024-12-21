#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from functions import sigmoid, add_zero_feature, unpack_params


def cost_function(
    nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef
):

    # 1. Распаковка параметров
    Theta1, Theta2 = unpack_params(
        nn_params, input_layer_size, hidden_layer_size, num_labels
    )

    # 2. Количество примеров
    m = X.shape[0]

    # 3. Прямое распространение для вычисления отклика сети

    # Входной слой (a1 = X, добавляем единичный столбец для нейрона смещения)
    A_1 = X  # уже добавлен нейрон смещения в главном файле
    Z_2 = np.dot(A_1, Theta1.T)  # линейная комбинация входов и весов
    A_2 = sigmoid(Z_2)  # активация скрытого слоя
    A_2 = add_zero_feature(A_2)  # добавляем единичный столбец для нейрона смещения

    # Выходной слой
    Z_3 = np.dot(A_2, Theta2.T)
    A_3 = sigmoid(Z_3)  # финальный отклик сети, это и есть H (h_theta)

    # 4. Ошибка (функция стоимости без регуляризации)
    term1 = -Y * np.log(A_3)  # -y * log(h_theta)
    term2 = (1 - Y) * np.log(1 - A_3)  # (1 - y) * log(1 - h_theta)
    J = (1 / m) * np.sum(term1 - term2)  # усреднение по примерам

    # 5. Регуляризация
    reg_Theta1 = np.sum(
        np.square(Theta1[:, 1:])
    )  # не включаем Theta1[:,0] (нейрон смещения)
    reg_Theta2 = np.sum(
        np.square(Theta2[:, 1:])
    )  # не включаем Theta2[:,0] (нейрон смещения)
    reg_J = (lambda_coef / (2 * m)) * (reg_Theta1 + reg_Theta2)  # регуляризатор

    # 6. Добавление регуляризации к функции стоимости
    J += reg_J

    return J
