#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Сигмоидная функция
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Производная сигмоидной функции
def sigmoid_derivative(z):
    g_z = sigmoid(z)
    return g_z * (1 - g_z)


# Построение графика сигмоиды
z = np.linspace(-10, 10, 1000)  # диапазон значений
sigmoid_values = sigmoid(z)

# Точки для касательных
points = [-1, -0.5, 0, 0.5, 1]
tangents = sigmoid_derivative(np.array(points))  # Производные в этих точках

# Настройка размера графика
plt.figure(figsize=(10, 6))  # Увеличенный размер графика

# Построение графика сигмоиды
plt.plot(z, sigmoid_values, label="Сигмоида", color="blue", linewidth=2)

# Проведение касательных
for point, tangent in zip(points, tangents):
    y_tangent = sigmoid(point) + tangent * (z - point)
    plt.plot(
        z,
        y_tangent,
        linestyle="--",
        linewidth=1.5,
        label=f"Касательная в {point}",
        alpha=0.8,
    )

# Отображение точек касательных
plt.scatter(
    points,
    sigmoid(np.array(points)),
    color="red",
    zorder=5,
    label="Точки касательных",
    s=10,
)

# Настройка графика
plt.xlabel("z", fontsize=14)
plt.ylabel("sigmoid(z)", fontsize=14)
plt.title("График сигмоиды и касательных в точках -1, -0.5, 0, 0.5, 1", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# Показать график
plt.show()


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
    Theta1 = nn_params[Theta1_start:Theta1_end].reshape(
        hidden_layer_size, input_layer_size + 1
    )

    Theta2_start = Theta1_end
    Theta2_end = Theta1_end + num_labels * (hidden_layer_size + 1)
    Theta2 = nn_params[Theta2_start:Theta2_end].reshape(
        num_labels, hidden_layer_size + 1
    )

    return Theta1, Theta2
