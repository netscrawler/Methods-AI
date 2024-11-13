import math
import matplotlib.pyplot as plt
import numpy as np

# Critics' ratings dictionary
critics = {
    'Кот Матроскин': {
        'Зима в Простоквашино': 2.5,
        'Каникулы в Простоквашино': 3.5,
        'Ёжик в тумане': 3.0,
        'Винни-Пух': 3.5,
        'Ну, погоди!': 2.5,
        'Котёнок по имени Гав': 3.0
    },
    'Пёс Шарик': {
        'Зима в Простоквашино': 3.0,
        'Каникулы в Простоквашино': 3.5,
        'Ёжик в тумане': 1.5,
        'Винни-Пух': 5.0,
        'Котёнок по имени Гав': 3.0,
        'Ну, погоди!': 3.5
    },
    'Почтальон Печкин': {
        'Зима в Простоквашино': 2.5,
        'Каникулы в Простоквашино': 3.0,
        'Винни-Пух': 3.5,
        'Котёнок по имени Гав': 4.0
    },
    'Корова Мурка': {
        'Каникулы в Простоквашино': 3.5,
        'Ёжик в тумане': 3.0,
        'Котёнок по имени Гав': 4.5,
        'Винни-Пух': 4.0,
        'Ну, погоди!': 2.5
    },
    'Телёнок Гаврюша': {
        'Зима в Простоквашино': 3.0,
        'Каникулы в Простоквашино': 4.0,
        'Ёжик в тумане': 2.0,
        'Винни-Пух': 3.0,
        'Котёнок по имени Гав': 3.0,
        'Ну, погоди!': 2.0
    },
    'Галчонок': {
        'Зима в Простоквашино': 3.0,
        'Каникулы в Простоквашино': 4.0,
        'Котёнок по имени Гав': 3.0,
        'Винни-Пух': 5.0,
        'Ну, погоди!': 3.5
    },
    'Дядя Фёдор': {
        'Каникулы в Простоквашино': 4.5,
        'Ну, погоди!': 1.0,
        'Винни-Пух': 4.0
    }
}

def sim_distance(critics, person1, person2):
    shared_items = {item for item in critics[person1] if item in critics[person2]}
    if not shared_items:
        return 0
    sum_of_squares = sum((critics[person1][item] - critics[person2][item]) ** 2 for item in shared_items)
    return 1 / (1 + math.sqrt(sum_of_squares))

def sim_pearson(critics, person1, person2):
    shared_items = {item for item in critics[person1] if item in critics[person2]}
    if not shared_items:
        return 0
    n = len(shared_items)
    sum1 = sum(critics[person1][item] for item in shared_items)
    sum2 = sum(critics[person2][item] for item in shared_items)
    sum1Sq = sum(critics[person1][item] ** 2 for item in shared_items)
    sum2Sq = sum(critics[person2][item] ** 2 for item in shared_items)
    pSum = sum(critics[person1][item] * critics[person2][item] for item in shared_items)
    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - sum1 ** 2 / n) * (sum2Sq - sum2 ** 2 / n))
    if den == 0:
        return 0
    return num / den

def top_matches(critics, person, n=5, similarity=sim_pearson):
    scores = [(similarity(critics, person, other), other) for other in critics if other != person]
    scores.sort(reverse=True)
    return scores[:n]

def plot_critics_ratings(critics, film1, film2):
    x = []
    y = []
    labels = []
    for critic, ratings in critics.items():
        if film1 in ratings and film2 in ratings:
            x.append(ratings[film1])
            y.append(ratings[film2])
            labels.append(critic)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]))
    plt.xlabel(f'{film1}')
    plt.ylabel(f'{film2}')
    plt.title(f'{film1} vs {film2}')
    plt.grid(True)
    plt.show()

def plot_similarity_with_best_fit(critics, person1, person2):
    shared_items = {item for item in critics[person1] if item in critics[person2]}
    if not shared_items:
        print("No shared items to compare.")
        return
    x = [critics[person1][item] for item in shared_items]
    y = [critics[person2][item] for item in shared_items]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    for i, item in enumerate(shared_items):
        plt.annotate(item, (x[i], y[i]))
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [m * xi + b for xi in x], color='red')
    plt.xlabel(f' {person1}')
    plt.ylabel(f'{person2}')
    plt.title(f'{person1} and {person2} ')
    plt.grid(True)
    plt.show()

def plot_top_matches(critics, person, n=5, similarity=sim_pearson):
    matches = top_matches(critics, person, n, similarity)
    scores, names = zip(*matches)
    plt.figure(figsize=(10, 6))
    plt.barh(names, scores, color='skyblue')
    plt.xlabel('')
    plt.title(f'Top {n} Critics Similar to {person}')
    plt.gca().invert_yaxis()
    plt.show()

# Plot for "Каникулы в Простоквашино" and "Ёжик в тумане"
plot_critics_ratings(critics, 'Каникулы в Простоквашино', 'Ёжик в тумане')

# Plot for Pearson Correlation with Best Fit Line between Шарик and Гаврюша
plot_similarity_with_best_fit(critics, 'Пёс Шарик', 'Телёнок Гаврюша')

# Plot for Pearson Correlation with Best Fit Line between Кот Матроскин and Галчонок
plot_similarity_with_best_fit(critics, 'Кот Матроскин', 'Галчонок')

# Plot the top matches for 'Кот Матроскин'
plot_top_matches(critics, 'Кот Матроскин')

# Example calls to sim_pearson
print(sim_pearson(critics, 'Кот Матроскин', 'Галчонок'))  # Example 1
print(sim_pearson(critics, 'Пёс Шарик', 'Телёнок Гаврюша'))  # Example 2
