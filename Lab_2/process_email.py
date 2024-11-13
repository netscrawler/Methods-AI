import numpy as np
from porterStemmer import porterStemmer
import re

def get_dictionary():
    with open('dictionary.txt') as f:
        dictionary = []
        for line in f:
            idx, w = line.split()
            dictionary.append(w)
    return np.array(dictionary)

def process_email(email):
    vocabList = get_dictionary()
    word_indices = []

    # приведение текста к нижнему регистру
    email = email.lower()

    # удаление html-тегов из текста письма
    rx = re.compile(r'<[^<>]+>|\n')
    email = rx.sub(' ', email)

    # числа заменяются на строку 'number'
    rx = re.compile(r'[0-9]+')
    email = rx.sub('number ', email)

    # ссылки заменяются на строку 'httpaddr'
    rx = re.compile(r'(http|https)://[^\s]*')
    email = rx.sub('httpaddr ', email)

    # электронные адреса заменяются на строку 'emailaddr'
    rx = re.compile(r'[^\s]+@[^\s]+')
    email = rx.sub('emailaddr ', email)

    # значок $ заменяется на строку 'dollar'
    rx = re.compile(r'[$]+')
    email = rx.sub('dollar ', email)

    # удаление не буквенно-цифровых символов
    rx = re.compile(r'[^a-zA-Z0-9 ]')
    email = rx.sub('', email).split()

    for str in email:
        # приведене каждого слова к существительному в единственном числе 
        try:
            str = porterStemmer(str.strip())
        except:
            str = ''
            continue

        if len(str) < 1:
            continue

        # Получение индекса слова в словаре
        ans = np.where(vocabList == str)
        if ans[0].size > 0:
            word_indices.append(ans[0][0])  # Добавляем индекс в word_indices

    return word_indices

def email_features(word_indices):
    n = 1899  # общее число слов в словаре
    x = np.zeros(n)

    # Устанавливаем в 1 элементы массива x, соответствующие индексам из word_indices.
    x[word_indices] = 1  # Устанавливаем элементы в 1 по индексам

    return x

