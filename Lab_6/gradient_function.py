import numpy as np
from functions import sigmoid, unpack_params, add_zero_feature
from sigmoid_gradient import sigmoid_gradient

def gradient_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    Theta1, Theta2 = unpack_params(nn_params, input_layer_size, hidden_layer_size, num_labels)
    m = X.shape[0]

    # Forward propagation
    A_1 = X  # A_1 имеет размер (5000, 401)
    Z_2 = A_1.dot(Theta1.T)     # Z_2 имеет размер (5000, 25)
    A_2 = add_zero_feature(sigmoid(Z_2))  # A_2 также должен иметь размер (5000, 26)
    Z_3 = A_2.dot(Theta2.T)     # Z_3 имеет размер (5000, num_labels)
    A_3 = sigmoid(Z_3)          # A_3 также имеет размер (5000, num_labels)

    # Backward propagation
    delta_3 = A_3 - Y           # delta_3 имеет размер (5000, num_labels)
    delta_2 = delta_3.dot(Theta2[:, 1:]) * sigmoid_gradient(Z_2)  # delta_2 имеет размер (5000, 25)

    Delta1 = delta_2.T.dot(A_1)  # Delta1 имеет размер (25, 401)
    Delta2 = delta_3.T.dot(A_2)  # Delta2 имеет размер (num_labels, 26)

    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m

    # Регуляризация
    Theta1_grad[:, 1:] += (lambda_coef / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lambda_coef / m) * Theta2[:, 1:]

    return np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

