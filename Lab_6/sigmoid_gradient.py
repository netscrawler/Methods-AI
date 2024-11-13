import numpy as np
from functions import sigmoid



def sigmoid_gradient(z):
    # Задание 4. Реализовать функцию вычисления производной сигмоиды в точке z
    sigmoid_z = sigmoid(z)  # Вычисляем значение сигмоиды
    derivative = sigmoid_z * (1 - sigmoid_z)  # Производная сигмоиды
    return derivative

