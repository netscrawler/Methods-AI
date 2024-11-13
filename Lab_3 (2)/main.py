import numpy as np
import matplotlib.pyplot as plt
import os

# Задание 1: Загрузка данных
def load_data(file_path):
    """Загрузка данных из файла."""
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, 0]  # Первый столбец
    y = data[:, 1]  # Второй столбец
    m = len(y)  # Количество примеров
    X = np.column_stack((np.ones(m), X))  # Добавляем единицы для свободного члена
    y = y.reshape(-1, 1)  # Приводим y к двумерному виду с одной колонкой
    return X, y

# Задание 2: Визуализация данных
def plot_data(X, y):
    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 1], y, 'b.', markersize=10)  # точечный график
    plt.title('Зависимость прибыльности от численности')
    plt.xlabel('Численность (10,000 чел.)')
    plt.ylabel('Прибыль (10,000 $)')
    plt.grid()
    plt.show()

# Задание 3: Вычисление функции стоимости
def compute_cost(X, y, theta):
    m = len(y)  # количество примеров
    h_x = X @ theta  # предсказанные значения
    cost = (1 / (2 * m)) * np.sum(np.power(h_x - y, 2))  # вычисление стоимости
    return cost

# Задание 4: Градиентный спуск
def gradient_descent(X, y, alpha, iterations):
    m = len(y)
    n = X.shape[1]
    theta = np.zeros((n, 1))  # инициализация коэффициентов
    J_theta = np.zeros((iterations, 1))  # для хранения значений стоимости

    for i in range(iterations):
        h_x = X @ theta  # предсказанные значения
        theta_temp = theta - (alpha / m) * (X.T @ (h_x - y))  # обновление коэффициентов
        theta = theta_temp
        J_theta[i] = compute_cost(X, y, theta)  # вычисление стоимости

    return theta, J_theta

# Задание 5: Предсказание
def predict(theta, X):
    return X @ theta

# Задание 6: Добавление линии регрессии на график
def plot_regression_line(X, y, theta):
    """Построение линии регрессии."""
    plt.scatter(X[:, 1], y.flatten(), color='blue', marker='o')  # точки данных
    x = np.arange(min(X[:, 1]), max(X[:, 1]), 0.1)  # значения для линии регрессии
    plt.plot(x, theta[0, 0] + theta[1, 0] * x, 'g--')  # линия регрессии
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Линейная регрессия')
    plt.savefig('regression_plot.png')  # Сохранение графика
    plt.close()  # Закрыть график, чтобы не возникали проблемы с отображением

# Основная часть: выполнение всех заданий
def main():
    """Основная функция."""
    X, y = load_data('ex1data1.txt')  # Разложение на X и y
    initial_cost = compute_cost(X, y, np.zeros((X.shape[1], 1)))  # передаем начальные значения theta
    print(f'Начальная стоимость: {initial_cost}')
    
    theta, _ = gradient_descent(X, y, alpha=0.01, iterations=1500)  # обновление theta
    print(f'Коэффициенты после градиентного спуска: {theta}')

    plot_regression_line(X, y, theta)

if __name__ == '__main__':
    main()

