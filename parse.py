import requests
from bs4 import BeautifulSoup
from collections import Counter

# URL страницы с цитатами
url = 'http://quotes.toscrape.com/'

# Выполнение GET-запроса к странице
response = requests.get(url)

# Проверка, успешно ли выполнен запрос
if response.status_code == 200:
    # Создание объекта BeautifulSoup для парсинга
    soup = BeautifulSoup(response.content, 'html.parser')

    # Поиск всех цитат на странице
    quotes = soup.find_all('div', class_='quote')

    # Список для хранения информации о цитатах
    quote_data = []
    authors = []  # Список для авторов

    for quote in quotes:
        # Извлечение текста цитаты
        text = quote.find('span', class_='text').text
        # Извлечение автора
        author = quote.find('small', class_='author').text
        quote_data.append({'text': text, 'author': author})
        authors.append(author)  # Добавляем автора в список

    # Вывод информации о цитатах
    for idx, quote in enumerate(quote_data):
        print(f"{idx + 1}. \"{quote['text']}\" - {quote['author']} (Длина: {len(quote['text'])})")

    # Вычисление средней длины цитат
    average_length = sum(len(quote['text']) for quote in quote_data) / len(quote_data)

    # Нахождение самой длинной и самой короткой цитаты
    longest_quote = max(quote_data, key=lambda x: len(x['text']))
    shortest_quote = min(quote_data, key=lambda x: len(x['text']))

    # Нахождение автора с самым большим количеством цитат
    author_count = Counter(authors)
    most_common_author, most_common_count = author_count.most_common(1)[0]

    # Вывод результатов
    print(f"\nСредняя длина цитат: {average_length:.2f} символов")
    print(f"Самая длинная цитата: \"{longest_quote['text']}\" - {longest_quote['author']} (Длина: {len(longest_quote['text'])})")
    print(f"Самая короткая цитата: \"{shortest_quote['text']}\" - {shortest_quote['author']} (Длина: {len(shortest_quote['text'])})")
    print(f"Количество уникальных авторов: {len(set(authors))}")
    print(f"Автор с самым большим количеством цитат: {most_common_author} ({most_common_count} цитат)")
else:
    print(f'Ошибка при получении страницы: {response.status_code}')

