import numpy as np
from porterStemmer import porterStemmer
import re
import scipy.io
from sklearn import svm
from collections import OrderedDict
from process_email import *
from porterStemmer import *
# Здесь вы вставляете функции get_dictionary, process_email и email_features

if __name__ == "__main__":
    # Задание 1: Загрузка и вывод текста письма
    print("\nЗадание 1: Загрузка и вывод текста письма\n")
    with open('email.txt', 'r') as file:
        email = file.read().replace('\n', '')
    print(email)

    # Задание 2: Предобработка письма
    print("\nЗадание 2: Предобработка письма\n")
    word_indices = process_email(email)
    print(*np.array(word_indices))

    print("\nЗадание 3: Преобразование в вектор признаков\n")
    features = email_features(word_indices)
    print("Длина вектора признаков:", len(features))
    print("Количество ненулевых элементов:", np.sum(features > 0))

    print("\nЗадание 4: Обучение классификатора\n")
    data = scipy.io.loadmat('train.mat')
    X = data['X']
    y = data['y'].flatten()
    print('Тренировка SVM-классификатора с линейным ядром...')
    clf = svm.SVC(C=0.1, kernel='linear', tol=1e-3)
    model = clf.fit(X, y)
    p = model.predict(X)
    accuracy = np.mean(p == y) * 100
    print('Точность на обучающей выборке:', accuracy)

    print("\nЗадание 5: Оценка точности на тестовой выборке\n")
    test_data = scipy.io.loadmat('test.mat')
    Xtest = test_data['Xtest']
    ytest = test_data['ytest'].flatten()
    p_test = model.predict(Xtest)
    test_accuracy = np.mean(p_test == ytest) * 100
    print('Точность на тестовой выборке:', test_accuracy)

    print("\nЗадание 6: Определение слов, чаще встречающихся в спаме\n")
    t = sorted(list(enumerate(model.coef_[0])), key=lambda e: e[1], reverse=True)
    d = OrderedDict(t)
    idx = list(d.keys())
    weight = list(d.values())
    dictionary = get_dictionary()  # Убедитесь, что эта функция доступна
    print('Топ-15 слов в письмах со спамом:')
    for i in range(15):
        print(' %-15s (%f)' % (dictionary[idx[i]], weight[i]))

