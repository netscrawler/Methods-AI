# Одномерная линейная регресси
# 1, 2 ЗАДАНИЕ
# Импорт необходимых библиотек

import matplotlib.pyplot as plt
import numpy as np

# Устанавливаем рабочую директорию

# Загрузка данных
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)  # Численность городов
y = data[:, 1].reshape(-1, 1)  # Прибыльность

# Добавление единичного столбца к X
m = X.shape[0]  # Количество элементов в Х
X_ones = np.hstack((np.ones((m, 1)), X))  # Добавляем столбец единиц

# Построение графика
plt.plot(X, y, 'b.')  # Точечный график
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')
plt.grid()
plt.show()

# 3 ЗАДАНИЕ
# Функция для вычисления стоимости


def compute_cost(X, y, theta):
    m = y.shape[0]
    h_x = X.dot(theta)
    error = h_x - y
    squared_error = np.power(error, 2)
    cost = (1 / (2 * m)) * np.sum(squared_error)
    return cost


# Инициализация коэффициентов
theta = np.matrix([1, 2]).T  # Вектор-столбец

# Вычисляем стоимость
cost_value = compute_cost(X_ones, y, theta)
print(f"Вычисленная стоимость: {cost_value}")  # Ожидаемое значение: 75.203

# 4 ЗАДАНИЕ
# Функция градиентного спуска


def gradient_descent(X, y, alpha, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Инициализация theta
    J_theta = np.zeros(iterations)  # Вектор для хранения значений стоимости

    for i in range(iterations):
        h_x = X.dot(theta)
        error = h_x - y
        theta -= (alpha / m) * (X.T.dot(error))  # Обновление значений theta
        J_theta[i] = compute_cost(X, y, theta)

    return theta, J_theta


print(f"Значения theta: {theta.flatten()}")

# Запуск градиентного спуска
alpha = 0.01


iterations = 500
theta, J_theta = gradient_descent(X_ones, y, alpha, iterations)

# Построение графика для J_theta
plt.plot(range(iterations), J_theta, 'r-')
plt.title('Снижение ошибки J(theta) с увеличением итераций')
plt.xlabel('Итерации')
plt.ylabel('Стоимость J(theta)')
plt.grid()
plt.show()

# 5 ЗАДАНИЕ
# Предположим, что у нас есть данные о численности населения для двух городов
# Пример численности населения для двух городов
new_cities = np.array([[1650], [3000]])

# Добавление единичного столбца для предсказания
new_cities_ones = np.hstack((np.ones((new_cities.shape[0], 1)), new_cities))

# Использование коэффициентов theta для вычисления предсказаний
predictions = new_cities_ones.dot(theta)

# Вывод предсказанных значений прибыли
for i in range(new_cities.shape[0]):
    print(
        f"Предсказанная прибыль для города с численностью {new_cities[i, 0]}: {predictions[i, 0]:.2f}")

# 6 ЗАДАНИЕ: Построение графика с линией регрессии
plt.plot(X, y, 'b.')  # Точечный график
plt.title('Зависимость прибыльности от численности')
plt.xlabel('Численность')
plt.ylabel('Прибыльность')

# Построение линии регрессии
x = np.arange(np.min(X), np.max(X), 1)  # Интервал для линии
plt.plot(x, theta[1] * x + theta[0], 'g')  # Линия регрессии

plt.grid()
plt.show()


# Многомерная линейная регрессия
# 7 ЗАДАНИЕ


# Устанавливаем рабочую директорию

# Загрузка данных
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]  # Площадь и количество комнат
y = data[:, 2].reshape(-1, 1)  # Цена дома, вектор-столбец

# Функция для нормализации данных


def feature_normalize(X):
    mean = np.mean(X, axis=0)  # Среднее значение по каждому столбцу
    std_dev = np.std(X, axis=0)  # Стандартное отклонение по каждому столбцу
    X_normalized = (X - mean) / std_dev  # Нормализация
    return X_normalized, mean, std_dev


# Нормализация входных данных
X_normalized, mean, std_dev = feature_normalize(X)

# Вывод результатов нормализации
print("Нормализованные данные:")
print(X_normalized)
print("Средние значения:")
print(mean)
print("Стандартные отклонения:")
print(std_dev)

# 8 ЗАДАНИЕ
# Добавление единичного столбца к X для свободного члена
m = X_normalized.shape[0]
X_ones = np.hstack((np.ones((m, 1)), X_normalized))  # Добавляем столбец единиц

# Функция для вычисления стоимости


def compute_cost(X, y, theta):
    m = y.shape[0]
    h_x = X.dot(theta)
    error = h_x - y
    squared_error = np.power(error, 2)
    cost = (1 / (2 * m)) * np.sum(squared_error)
    return cost

# Функция градиентного спуска


def gradient_descent(X, y, alpha, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Инициализация theta
    J_theta = np.zeros(iterations)  # Вектор для хранения значений стоимости

    for i in range(iterations):
        h_x = X.dot(theta)
        error = h_x - y
        theta -= (alpha / m) * (X.T.dot(error))  # Обновление значений theta
        J_theta[i] = compute_cost(X, y, theta)

    return theta, J_theta


# Запуск градиентного спуска
alpha = 0.02
iterations = 500
theta, J_theta = gradient_descent(X_ones, y, alpha, iterations)

print(f"Значения theta: {theta.flatten()}")

# Прогнозирование для двух квартир
# Примерные данные о двух квартирах
new_apartments = np.array([[165, 3], [300, 4]])
new_apartments_normalized = (new_apartments - mean) / std_dev  # Нормализация
# Cтолбец единиц к нормализованному массиву
new_apartments_ones = np.hstack(
    (np.ones((new_apartments_normalized.shape[0], 1)), new_apartments_normalized))

# Использование коэффициентов theta для вычисления предсказаний
predictions = new_apartments_ones.dot(theta)

# Вывод предсказанных цен
for i in range(new_apartments.shape[0]):
    print(
        f"Предсказанная цена для квартиры с площадью {new_apartments[i, 0]} и количеством комнат {new_apartments[i, 1]}: {predictions[i, 0]:.2f}")


# Метод наименьших квадратов(МНК)
# Устанавливаем рабочую директорию

# Загрузка данных
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]  # Площадь и количество комнат
y = data[:, 2].reshape(-1, 1)  # Цена дома

# Добавление единичного столбца к X
m = X.shape[0]
X_ones = np.hstack((np.ones((m, 1)), X))  # Добавляем столбец единиц

# Метод наименьших квадратов (МНК)


def ordinary_least_squares(X, y):
    # Вычисляем коэффициенты theta
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


# Получение коэффициентов с помощью МНК
theta_ols = ordinary_least_squares(X_ones, y)

print(f"Значения theta (МНК): {theta_ols.flatten()}")

# Прогнозирование для двух квартир
# Примерные данные о двух квартирах
new_apartments = np.array([[1650, 3], [3000, 4]])
# Cтолбец единиц к нормализованному массиву
new_apartments_ones = np.hstack(
    (np.ones((new_apartments.shape[0], 1)), new_apartments))

# Использование коэффициентов theta для вычисления предсказаний
predictions_ols = new_apartments_ones.dot(theta_ols)

# Вывод предсказанных цен
for i in range(new_apartments.shape[0]):
    print(
        f"Предсказанная цена для квартиры с площадью {new_apartments[i, 0]} и количеством комнат {new_apartments[i, 1]} (МНК): {predictions_ols[i, 0]:.2f}")
