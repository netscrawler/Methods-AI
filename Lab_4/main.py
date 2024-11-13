import numpy as np
import random
import os

from numpy._core.multiarray import datetime_data
# Задание 1: Чтение условий задачи из файла


def read_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
        N = int(float(data[0].strip()))  # Преобразование в float, затем в int
        matrix = np.array([[int(float(num)) for num in line.strip().split()]
                          # Преобразование элементов в int
                           for line in data[1:N + 1]])
    return N, matrix

# Задание 2: Функция оценки стоимости пути


def calculate_cost(path, matrix):
    cost = 0
    for i in range(len(path)):
        cost += matrix[path[i]][path[(i + 1) % len(path)]]
    return cost

# Задание 3: Функция мутации


def mutate(path):
    path = path.copy()
    # Случайно выбираем город
    city_to_move = random.choice(path)
    # Убираем его из текущей позиции
    path.remove(city_to_move)
    # Случайно выбираем новое место для вставки
    new_position = random.randint(0, len(path))
    # Вставляем город на новое место
    path.insert(new_position, city_to_move)
    return path

# Задание 4: Функция кроссинговера (жадная стратегия)


def crossover(parent1, parent2, matrix):
    start_city = random.choice(parent1)
    child = [start_city]
    visited = {start_city}
    current_parent = 1  # Начинаем с первого родителя
    current_city = start_city

    while len(child) < len(parent1):
        neighbors = []
        # Находим соседние города в текущем родителе
        for index, city in enumerate(parent1):
            if city == current_city:
                # Добавляем соседние города
                if index > 0 and parent1[index - 1] not in visited:
                    neighbors.append(parent1[index - 1])
                if index < len(parent1) - 1 and parent1[index + 1] not in visited:
                    neighbors.append(parent1[index + 1])
                break

        if not neighbors:
            break  # Если нет соседей, выходим из цикла

        # Если есть непосещенные соседи, выбираем ближайшего
        unvisited_neighbors = [
            city for city in neighbors if city not in visited]
        if unvisited_neighbors:
            next_city = min(unvisited_neighbors,
                            key=lambda city: matrix[current_city][city])
            child.append(next_city)
            visited.add(next_city)
            current_city = next_city
        else:
            # Если все соседи посещены, выбираем ближайший город из второго родителя
            current_parent = 2 if current_parent == 1 else 1
            if current_parent == 2:
                neighbors = []
                for index, city in enumerate(parent2):
                    if city == current_city:
                        if index > 0 and parent2[index - 1] not in visited:
                            neighbors.append(parent2[index - 1])
                        if index < len(parent2) - 1 and parent2[index + 1] not in visited:
                            neighbors.append(parent2[index + 1])
                        break

                # Выбираем ближайшего из непосещенных соседей
                unvisited_neighbors = [
                    city for city in neighbors if city not in visited]
                if unvisited_neighbors:
                    next_city = min(unvisited_neighbors,
                                    key=lambda city: matrix[current_city][city])
                    child.append(next_city)
                    visited.add(next_city)
                    current_city = next_city

    return child

# Задание 5: Генерация начальной популяции


def generate_initial_population(size, N):
    population = []
    for _ in range(size):
        path = list(range(N))
        random.shuffle(path)
        population.append(path)
    return population

# Задание 6: Генетический алгоритм


def genetic_algorithm(N, matrix, population_size=100, generations=500, mutation_rate=0.1):
    population = generate_initial_population(population_size, N)
    best_cost = float('inf')
    best_path = None

    for generation in range(generations):
        # Evaluate costs
        costs = [(calculate_cost(path, matrix), path) for path in population]
        costs.sort(key=lambda x: x[0])  # Sort by cost
        population = [path for _, path in costs]  # Keep the best paths

        # Update best solution
        if costs[0][0] < best_cost:
            best_cost = costs[0][0]
            best_path = costs[0][1]
            print(
                f"Генерация {generation + 1}: Лучшая стоимость = {best_cost}, Путь = {best_path}")

        # Create new population
        new_population = population[:int(
            population_size * 0.1)]  # Elitism: keep the best 10%

        # Crossover and mutation
        while len(new_population) < population_size:
            # Select parents from the best half
            parent1, parent2 = random.choices(population[:50], k=2)
            child = crossover(parent1, parent2, matrix)

            # Mutation
            if random.random() < mutation_rate:
                mutated_child = mutate(child)
                new_population.append(mutated_child)
            else:
                new_population.append(child)

        population = new_population

    return best_path, best_cost


# Пример использования
if __name__ == '__main__':
    for i in range(1, 11):  # От var1.txt до var10.txt
        print("="*40)
        filename = f'var{i}.txt'  # Формирование имени файла
        try:
            N, matrix = read_data(filename)
            best_path, best_cost = genetic_algorithm(N, matrix)

            # Вывод конечных результатов
            print(f"\nРезультаты для файла {filename}:")
            print(f"Количество городов:", N)
            print(f"Матрица стоимости путей:\n", matrix)
            print("Лучший путь:", best_path)
            print("Лучшая стоимость:", best_cost)

        except FileNotFoundError:
            print(f"Файл {filename} не найден.")
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")
