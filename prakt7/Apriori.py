import numpy as np


# функция генерирует из k-1 элементных кандидатов k-элементных кандидатов
# Fk_1 - набор кандидатов с предыдущего этапа. Набор кандидатов представляет
# собой матрицу, каждая строка которой соответствует отдельному кандидату.
# Кандидат - это набор событий, который нужно будет проверить, превышает ли
# его поддержка порог или нет. Например, кандидат 1 2 4 означает, что нужно
# проверить поддержку для сочетания из первого, второго и четвертого событий.
# Fk - новые кандидаты для проверки с размером на 1 больше
def generate_candidates(previous_candidates, unique_elements):

    if len(previous_candidates) == 0:
        return []

    previous_candidate_size = previous_candidates.shape[1]
    next_candidates = np.array([], dtype=int)

    for candidate in previous_candidates:
        candidate_max_element = max(candidate)
        greater_elements = unique_elements[unique_elements >
                                           candidate_max_element]
        for element in greater_elements:
            next_candidates = np.append(
                next_candidates, np.append(candidate, element))

    return next_candidates.reshape(-1, previous_candidate_size + 1)


# вычисление поддержки кандидата
# X - исходная матрица с наблюдениями
# candidate - сочетание, для которого нужно посчитать поддержку
# candidate может иметь вид, например, [1 3] - это означает сочетание из первого и третьего событий
# Функия support должна посчитать, как часто данное сочетание встречается в матрице X
# Например, пусть матрица X имеет вид:
# a b c
# b c
# a b d
# сочетание [a b] встречается 2 раза из трех транзакций, поэтому его поддержка sup равна 2/3 = 0.67
def support(X, candidate):
    return np.mean(np.all(X[:, candidate] == 1, axis=1))
    # TODO: реализуйте вычисление поддержки для кандидата candidate.
    # воспользуйтесь функциями mean и all из состава библиотеки numpy
    # return ...


# функция поиска ассоциативных правил с помощью алгоритма Apriori.
# X представляет собой матрицу, в которой каждая строка описывает
# отдельную транзакцию. Столбцы означают события, произошедшие в
# этой транзакции. Если событие произошло, ему соответствует элемент
# в матрице X со значением 1, если не произошло - со значением 0.
# sup - минимальная поддержка - число в диапазоне от 0 до 1.
# Например, sup = 0.5 означает, что будут найдены сочетания событий,
# встречающиеся в 50% транзакций и более.
#
# функция возвращает матрицу rules
# Каждая строка в матрице rules - это отдельное ассоциативное правило.
# Каждая строка имеет примерно такой вид: [1 0 1 0 0.5]
# В этой строке закодировано сочетание событий (например, как здесь:
# первое и третье событие происходят одновременно) и поддержка для
# этого сочетания - 0.5 (последний элемент в строке).
def apriori(X, minimal_support):
    rules = []
    support_ = np.mean(X, axis=0)
    # TODO: посчитайте поддержку для каждого 1-элементного кандидата
    # Например, если матрица X имеет вид:
    # 1 0 1 1
    # 0 1 0 1
    # 0 1 0 0
    # то поддержка для каждого события (каждый столбец в X соответствует событию) будет:
    # [1/3 2/3 1/3 2/3] или [0.33 0.67 0.33 0.67]
    # сохраните результат в вектор-строку support_, длина которого равна 4
    # (количество столбцов в X = количество наблюдаемых событий)
    # support_ = ...   # поддержка каждого 1-элементного набора

    n = X.shape[1]                  # количество наблюдаемых событий

    # часто встречающиеся 1-элементные наборы
    # one_element_candidates будет содержать номера событий, которые
    # встречаются чаще, чем минимальная поддержка minimal_support.
    one_element_candidates = np.where(support_ > minimal_support)[
        0].reshape(-1, 1)
    unique_elements = np.sort(np.unique(one_element_candidates)).reshape(-1, 1)

    accepted_candidates = one_element_candidates

    # необходимо рассмотреть все возможные наборы от двухэлементных до n-элементных
    for k in range(2, n+1):
        multi_element_candidates = generate_candidates(
            accepted_candidates, unique_elements)    # генерация кандидатов
        # в accepted_candidates будут сохранены только кандидаты, поддержка для которых превысила порог
        accepted_candidates = np.array([], dtype=int)

        # вычисление поддержки для каждого кандидата. Лишние кандидаты отбрасываются
        for candidate in multi_element_candidates:
            if support(X, candidate) > minimal_support:
                accepted_candidates = np.append(
                    accepted_candidates, candidate).reshape(-1, k)
                # TODO: если i-й кандидат из multi_element_candidates прошел проверку, то есть поддержка для него
                # оказалась выше минимальной поддержки, значит необходимо внести его в матрицу
                # accepted_candidates для того, чтобы на его основе генерировать новых кандидатов с размером на 1 больше
                # добавьте i-го кандидата из multi_element_candidates как новую строку в матрицу accepted_candidates
                # используйте функцию numpy.reshape для формирования матрицы accepted_candidates нужной размерности
                # accepted_candidates = np.append(...).reshape(...)

                # вносим кандидата, прошедшего порог, в набор ассоциативных правил
                sub = np.zeros((1, n))
                sub[:, candidate] = 1
                sub = np.append(sub, support(X, candidate))
                rules = np.append(rules, sub).reshape(-1, len(sub))

    return rules
