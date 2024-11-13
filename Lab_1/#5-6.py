import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from svm import visualize_boundary_linear, partial, gaussian_kernel, svm_train, visualize_boundary

# Задание 5
data = sio.loadmat('C:/Users/semen/Downloads/Lab_1/Lab_1/dataset2.mat')
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