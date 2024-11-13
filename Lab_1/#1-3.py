import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from svm import svm_train, linear_kernel, visualize_boundary_linear

# Задание 1
data = sio.loadmat('C:/Users/semen/Downloads/Lab_1/Lab_1/dataset1.mat')
X = data['X']
y = data['y'].astype(np.float64)

# Задание 2 и 3
C = 100
model = svm_train(X, y, C, linear_kernel)

plt.figure()
visualize_boundary_linear(X, y, model, 'Вывод графика')
plt.show()
