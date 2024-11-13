#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Открытие изображения и преобразование его в массив NumPy
im = Image.open('image.jpg')  # Укажите путь к изображению
data = np.array(im.getdata()).reshape([im.height, im.width, 3])

# Функция для сжатия строки изображения с использованием полиномиальной регрессии


def compress_row(y, degree=5, bits_per_channel=4):
    # Создаем матрицу входных признаков
    x = np.arange(len(y))
    X = np.array([x**i for i in range(1, degree + 1)]).transpose()

    # Применяем полиномиальную регрессию
    lm = linear_model.LinearRegression()
    lm.fit(X, y)
    predicted = lm.predict(X)

    # Вычисляем разности и кодируем их
    diff = y - predicted
    threshold = 2**(bits_per_channel - 1) - 1
    diff_clipped = np.clip(diff, -threshold, threshold)

    # Восстанавливаем значения после сжатия
    y_compressed = np.clip(predicted + diff_clipped, 0, 255).astype(np.uint8)
    return y_compressed


# Применение сжатия ко всему изображению
bits = [3, 4, 5, 6, 7]  # Биты для различных вариантов сжатия
compressed_images = []

for b in bits:
    # Создаем копию изображения для каждого числа бит
    im_copy = im.copy()
    pix = im_copy.load()

    # Обрабатываем каждый канал и каждую строку изображения
    for row in range(im.height):
        for channel in range(3):  # RGB каналы
            y = data[row, :, channel]
            y_compressed = compress_row(y, degree=5, bits_per_channel=b)

            # Обновляем пиксели в строке для данного канала
            for col in range(im.width):
                l = list(pix[col, row])
                l[channel] = int(y_compressed[col])
                pix[col, row] = tuple(l)

    # Сохраняем результат сжатия
    compressed_images.append(im_copy)
    im_copy.save(f'ready_{b}_bits.png')

# Визуализация изображений
fig, axs = plt.subplots(1, len(bits) + 1, figsize=(15, 5))
axs[0].imshow(im)
axs[0].set_title("Original")

for i, b in enumerate(bits):
    axs[i + 1].imshow(compressed_images[i])
    axs[i + 1].set_title(f"{b} bits")

plt.show()
