from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
from functions import (
    add_zero_feature,
    sigmoid,
    decode_y,
    rand_initialize_weights,
    pack_params,
    unpack_params,
)
from cost_function import cost_function
from sigmoid_gradient import sigmoid_gradient
from gradient_functions import gradient_function

if __name__ == "__main__":
    # ЗАДАНИЕ 1
    # Загрузка обучающей выборки
    data = loadmat("training_set.mat")
    X = data["X"]  # признаки
    y = data["y"]  # метки классов

    # Загрузка весовых коэффициентов
    weights = loadmat("weights.mat")
    Theta1 = weights["Theta1"]
    Theta2 = weights["Theta2"]

    # Проверка размеров загруженных данных
    print(f"Размер X: {X.shape}")  # Ожидаемый размер: (5000, 400)
    print(f"Размер y: {y.shape}")  # Ожидаемый размер: (5000, 1)
    print(f"Размер Theta1: {Theta1.shape}")  # Ожидаемый размер: (25, 401)
    print(f"Размер Theta2: {Theta2.shape}")  # Ожидаемый размер: (10, 26)

    # Загрузить обучающую выборку из файла training_set.mat в переменные X и y
    # Загрузить весовые коэффициенты из файла weights.mat в переменные Theta1 и Theta2
    # Использовать для этого функцию scipy.io.loadmat

    # ЗАДАНИЕ 2
    # Программно определить параметры нейронной сети
    # input_layer_size = ...  # количество входов сети (20*20=400)
    # hidden_layer_size = ... # нейронов в скрытом слое (25)
    # num_labels = ...        # число распознаваемых классов (10)
    # m = ...                 # количество примеров (5000)

    input_layer_size = 400  # 20x20 пикселей
    hidden_layer_size = 25  # 25 нейронов в скрытом слое
    num_labels = 10  # 10 классов
    m = X.shape[0]  # количество примеров

    # добавление единичного столбца - нейрон смещения
    X = add_zero_feature(X)

    # декодирование вектора Y
    Y = decode_y(y)

    # объединение матриц Theta в один большой массив
    nn_params = pack_params(Theta1, Theta2)

    # ЗАДАНИЕ 3
    # проверка функции стоимости для разных lambda
    lambda_coef = 0
    print(
        "Функция стоимости для lambda {} = {}".format(
            lambda_coef,
            cost_function(
                nn_params,
                input_layer_size,
                hidden_layer_size,
                num_labels,
                X,
                Y,
                lambda_coef,
            ),
        )
    )

    lambda_coef = 1
    print(
        "Функция стоимости для lambda {} = {}".format(
            lambda_coef,
            cost_function(
                nn_params,
                input_layer_size,
                hidden_layer_size,
                num_labels,
                X,
                Y,
                lambda_coef,
            ),
        )
    )

    # ЗАДАНИЕ 4
    # проверка производной sigmoid
    gradient = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print("Производная функции sigmoid в точках -1, -0.5, 0, 0.5, 1:")
    print(gradient)

    # случайная инициализация параметров
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = pack_params(initial_Theta1, initial_Theta2)

    # ЗАДАНИЕ 5
    # обучение нейронной сети
    res = minimize(
        cost_function,
        initial_nn_params,
        method="L-BFGS-B",
        jac=gradient_function,
        options={"maxiter": 100},
        args=(input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef),
    ).x

    # разбор вычисленных параметров на матрицы Theta1 и Theta2
    Theta1, Theta2 = unpack_params(res, input_layer_size, hidden_layer_size, num_labels)

    # выичисление отклика сети на примеры из обучающей выборки
    h1 = sigmoid(np.dot(X, Theta1.T))
    h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))
    y_pred = np.argmax(h2, axis=1) + 1

    print(
        "Точность нейронной сети на обучающей выборке: {}".format(
            np.mean(
                y_pred == y.ravel(),
            )
            * 100
        )
    )
