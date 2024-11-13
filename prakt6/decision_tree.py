import numpy as np
import math

NOT_FIT = 1         # кандидатка не подходит
FIT = 2             # кандидатка подходит

NUMERICAL = 0       # параметр числовой
CATEGORICAL = 1     # параметр категориальный

def decision_tree(X, Y, scale, level=0):
    # Проверка, является ли узел листом
    if len(np.unique(Y)) == 1:
        print('class = %d' % Y[0])
        return

    print('')

    n = X.shape[1]  # количество признаков
    m = X.shape[0]  # количество примеров

    # Энтропия до разбиения
    info = Info(Y)

    gain = []
    thresholds = np.zeros(n)

    # Цикл вычисления информационного выигрыша по каждому столбцу выборки
    for i in range(n):
        if scale[i] == CATEGORICAL:   # категориальный признак
            unique_values = np.unique(X[:, i])
            info_s = 0

            for value in unique_values:
                subset_y = Y[X[:, i] == value]
                probability = len(subset_y) / m
                info_s += probability * Info(subset_y)

            gain.append(info - info_s)
        
        else:  # непрерывный признак
            # Сортируем столбец по возрастанию
            val = np.sort(X[:, i])
            local_gain = np.zeros(m - 1)

            # Количество порогов на 1 меньше числа примеров
            for j in range(m - 1):
                threshold = val[j]
                less = sum(X[:, i] <= threshold)  # Количество значений в столбце <= порога
                greater = m - less  # Количество значений в столбце > порога

                # Вычисляем информативность признака при данном пороге
                info_s = (less / m) * Info(Y[X[:, i] <= threshold]) + (greater / m) * Info(Y[X[:, i] > threshold])
                local_gain[j] = info - info_s

            gain.append(np.max(local_gain, axis=0))
            idx = np.argmax(local_gain, axis=0)
            thresholds[i] = val[idx]

    # Теперь нужно выбрать столбец с максимальным приростом информации
    max_idx = np.argmax(gain)

    if scale[max_idx] == CATEGORICAL:
        # Если этот столбец категориальный
        categories = np.unique(X[:, max_idx])

        for category in categories:
            # Рекурсивно вызываем функцию decision_tree
            sub_x = X[X[:, max_idx] == category, :]
            sub_y = Y[X[:, max_idx] == category]

            print_indent(level)
            print('column %d == %f, ' % (max_idx, category), end='')

            decision_tree(sub_x, sub_y, scale, level + 1)
    else:
        # Столбец числовой
        threshold = thresholds[max_idx]

        # Рекурсивно вызываем decision_tree для значений меньше порога
        sub_x = X[X[:, max_idx] <= threshold, :]
        sub_y = Y[X[:, max_idx] <= threshold]

        print_indent(level)
        print('column %d <= %f, ' % (max_idx, threshold), end='')

        decision_tree(sub_x, sub_y, scale, level + 1)

        # Рекурсивно вызываем decision_tree для значений больше порога
        sub_x = X[X[:, max_idx] > threshold, :]
        sub_y = Y[X[:, max_idx] > threshold]

        print_indent(level)
        print('column %d >  %f, ' % (max_idx, threshold), end='')

        decision_tree(sub_x, sub_y, scale, level + 1)

# Вычисление энтропии множества set
def Info(set):
    m = len(set)
    info = 0

    # Получаем уникальные классы и их частоты
    unique_classes, counts = np.unique(set, return_counts=True)

    # Рассчитываем энтропию
    for count in counts:
        probability = count / m
        if probability > 0:
            info -= probability * np.log2(probability)

    return info

# Печать отступа для наглядности
def print_indent(level):
    print(level * '  ', end='')

