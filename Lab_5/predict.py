import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    a1 = np.c_[ones, X]

    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones((a2.shape[0], 1)), a2]

    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    return np.argmax(a3, axis=1) + 1