import numpy as np
from random import shuffle, randint
import math

# k - количество искомых кластеров
# X - объекты
def k_means(k, X):
    m = X.shape[0]  # количество точек
    n = X.shape[1]  # количество признаков объекта (например, рост и вес)
    curr_iteration = prev_iteration = np.zeros([m, 1])

    # Генерируем k случайных центров из данных X
    idx = np.arange(0, m)
    shuffle(idx)
    centers = X[idx[:k], :]

    # Создаем массив для хранения всех центров на каждом шаге
    all_centers = np.copy(centers)
    errors = np.array([])

    # Приписываем каждую точку к ближайшему кластеру и вычисляем ошибку
    curr_iteration, e = class_of_each_point(X, centers)
    errors = np.append(errors, e)

    # Цикл до тех пор, пока центры не стабилизируются
    iteration_count = 1
    while not np.all(curr_iteration == prev_iteration):
        prev_iteration = curr_iteration

        # Пересчитываем центры кластеров
        for i in range(k):
            sub_X = X[curr_iteration == i, :]
            if len(sub_X) > 0:
                centers[i, :] = mass_center(sub_X)
            else:
                centers[i, :] = X[randint(0, m-1)]

        # Добавляем текущие центры кластеров в all_centers
        all_centers = np.append(all_centers, centers, axis=0)

        # Приписываем каждую точку к ближайшему центру и вычисляем ошибку
        curr_iteration, e = class_of_each_point(X, centers)
        errors = np.append(errors, e)

        iteration_count += 1

    # Преобразуем all_centers в трехмерный массив для удобства отображения траекторий
    all_centers = np.reshape(all_centers, (iteration_count, k, n))

    # Возвращаем центры кластеров, все центры на каждой итерации и ошибки
    return centers, all_centers, errors


# Вычисление расстояния между двумя точками
def dist(p1, p2):
    return math.sqrt(sum((p1 - p2)**2))


# Вычисление центра масс для набора точек (среднее по каждому столбцу)
def mass_center(X):
    return np.mean(X, axis=0)


# Определение класса (кластера) для каждой точки и вычисление ошибки
def class_of_each_point(X, centers):
    m = X.shape[0]
    k = centers.shape[0]

    # Матрица расстояний от каждой точки до каждого центра
    distances = np.zeros([k, m])
    for i in range(k):
        for j in range(m):
            distances[i, j] = dist(centers[i], X[j])

    # Нахождение ближайшего центра для каждой точки
    min_dist = np.min(distances, axis=0)
    classes = np.argmin(distances, axis=0)

    # Вычисляем ошибку кластеризации как средний квадрат расстояния
    err = np.mean(pow(min_dist, 2))

    return classes, err

