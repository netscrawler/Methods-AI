import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from svm import svm_train, linear_kernel, visualize_boundary_linear
from svm import visualize_boundary_linear, partial, gaussian_kernel, svm_train, visualize_boundary, svm_predict
import svm


# Задание 1
data = sio.loadmat('dataset1.mat')
X = data['X']
y = data['y'].astype(np.float64)

# Задание 2 и 3
C = 100
model = svm_train(X, y, C, linear_kernel)

plt.figure()
visualize_boundary_linear(X, y, model, 'Вывод графика')
plt.show()

svm.contour(1)
svm.contour(3)


data = sio.loadmat('dataset2.mat')
X = data['X']
y = data['y'].astype(np.float64)

C = 1.0
sigma =0.1
gaussian = partial(gaussian_kernel, sigma=sigma)
gaussian.__name__ = gaussian_kernel.__name__
model = svm_train(X, y, C, gaussian)

plt.figure()
visualize_boundary(X, y, model, title='График')
plt.show()

data = sio.loadmat('dataset3.mat')
X = data['X']
y = data['y'].astype(np.float64)
Xval = data['Xval']
yval = data['yval'].astype(np.float64)

C =1.0
sigma =0.5
gaussian = partial(gaussian_kernel, sigma=sigma)
gaussian.__name__ = gaussian_kernel.__name__
model = svm_train(X, y, C, gaussian)

plt.figure()
visualize_boundary(X, y, model, title='График')
plt.show()

plt.figure()
visualize_boundary_linear(Xval, yval, None, 'Тестовой выборки')
plt.show()
# ------------------------
# Задаем возможные значения C и sigma
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

best_C = None
best_sigma = None
min_error = float('inf')

for C in C_values:
    for sigma in sigma_values:
        # Определение гауссового ядра с текущим sigma
        gaussian = partial(gaussian_kernel, sigma=sigma)
        gaussian.__name__ = gaussian_kernel.__name__

        # Обучение модели с текущими значениями C и sigma
        model = svm_train(X, y, C, gaussian)

        # Предсказание для тестовой выборки
        predictions = svm_predict(model, Xval)

        # Вычисление ошибки предсказания
        error = np.mean(predictions.ravel() != yval.ravel())

        # Сравнение и сохранение наилучших параметров
        if error < min_error:
            min_error = error
            best_C = C
            best_sigma = sigma

# Вывод наилучших значений C и sigma
print(f'Оптимальные параметры: C = {best_C}, sigma = {best_sigma}')

# Обучение модели с оптимальными параметрами
optimal_gaussian = partial(gaussian_kernel, sigma=best_sigma)
optimal_gaussian.__name__ = gaussian_kernel.__name__
optimal_model = svm_train(X, y, best_C, optimal_gaussian)

# Визуализация наилучшей модели для обучающей выборки
plt.figure()
visualize_boundary(X, y, optimal_model, title='Обучающая выборка - наилучшая модель')
plt.show()

# Визуализация наилучшей модели для тестовой выборки
plt.figure()
visualize_boundary(Xval, yval, optimal_model, title='Тестовая выборка - наилучшая модель')
plt.show()

