#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(z):
    """Вычисляет значение сигмоиды."""
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    """Вычисляет производную сигмоиды."""
    sigmoid_z = sigmoid(z)  # Вычисляем значение сигмоиды
    return sigmoid_z * (1 - sigmoid_z)  # Производная сигмоиды

def add_zero_feature(X):
    """Добавляет единичный столбец (нейрон смещения) к матрице X."""
    return np.hstack((np.ones((X.shape[0], 1)), X))

def decode_y(y):
    """Преобразует вектор меток в формат one-hot."""
    m = y.size
    num_labels = np.max(y)
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] - 1] = 1  # Предполагается, что метки от 1 до num_labels
    return Y

def rand_initialize_weights(L_in, L_out):
    """Случайная инициализация весов."""
    epsilon_init = 0.12  # Меньшее значение, чтобы избежать симметрии
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init

def pack_params(Theta1, Theta2):
    """Упаковывает матрицы Theta в один вектор."""
    return np.concatenate([Theta1.ravel(), Theta2.ravel()])

def unpack_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    """Разбирает вектор nn_params на матрицы Theta1 и Theta2."""
    Theta1_start = 0
    Theta1_end = hidden_layer_size * (input_layer_size + 1)
    Theta1 = nn_params[Theta1_start:Theta1_end].reshape(hidden_layer_size, input_layer_size + 1)

    Theta2_start = Theta1_end
    Theta2_end = Theta1_end + num_labels * (hidden_layer_size + 1)
    Theta2 = nn_params[Theta2_start:Theta2_end].reshape(num_labels, hidden_layer_size + 1)

    return Theta1, Theta2

