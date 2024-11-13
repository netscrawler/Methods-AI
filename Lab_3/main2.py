from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

# Шаг 1: Загрузка данных
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

# Преобразуем X и y в матрицы
X = np.matrix(X).T
y = np.matrix(y).T

# Шаг 2: Построение графика зависимость прибыли от численности
font = {'family': 'sans-serif', 'weight': 'normal'}
rc('font', **font)

plt.figure()
plt.plot(X, y, 'b.')
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность (в 10 000 чел.)')
plt.ylabel('Прибыльность (в 10 000)')
plt.grid(True)
plt.show()  # Сохраняем график в файл
# Закрываем фигуру

# Шаг 3: Добавление столбца единиц к X
X_ones = np.c_[np.ones((m, 1)), X]

# Шаг 4: Функция стоимости


def compute_cost(X, y, theta):
    m = len(y)
    h_x = X * theta
    cost = (1 / (2 * m)) * np.sum(np.power(h_x - y, 2))
    return cost

# Шаг 5: Градиентный спуск


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h_x = X * theta
        theta = theta - (alpha / m) * (X.T * (h_x - y))
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


# Шаг 6: Применение градиентного спуска
alpha = 0.01  # шаг градиентного спуска
num_iters = 1500  # количество итераций
theta = np.matrix(np.zeros((2, 1)))  # Инициализация theta в 0
theta, J_history = gradient_descent(X_ones, y, theta, alpha, num_iters)
#
#  # Выводим оптимальные theta
print(f"Оптимальные параметры theta:\n{theta}")

# Выводим значение функции стоимости для этих theta
final_cost = compute_cost(X_ones, y, theta)
print(f"Финальная стоимость: {final_cost:.4f}")

# Шаг 7: Построение линии регрессии
plt.figure()
plt.plot(X, y, 'b.')  # Исходные данные
plt.plot(X, X_ones * theta, 'r-')  # Линия регрессии
plt.title('Линейная регрессия')
plt.xlabel('Численность (в 10 000 чел.)')
plt.ylabel('Прибыльность (в 10 000)')
plt.grid(True)
plt.show()  # Сохраняем график в файл
# Закрываем фигуру

# Шаг 8: Построение графика функции стоимости
plt.figure()
plt.plot(np.arange(num_iters), J_history, 'b-')
plt.title('Функция стоимости')
plt.xlabel('Итерации')
plt.ylabel('Стоимость J')
plt.grid(True)
plt.show()  # Сохраняем график в файл
# Закрываем фигуру

# Шаг 9: Прогноз
population = 3.5  # Прогнозируем для численности 35 000
prediction = np.matrix([1, population]) * theta
print(
    f"Прогнозируемая прибыль для численности 35 000: {prediction[0, 0] * 10000:.2f}")

population = 7.0  # Прогнозируем для численности 70 000
prediction = np.matrix([1, population]) * theta
print(
    f"Прогнозируемая прибыль для численности 70 000: {prediction[0, 0] * 10000:.2f}")
