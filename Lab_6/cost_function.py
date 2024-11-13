import numpy as np
from functions import sigmoid, unpack_params, add_zero_feature

def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    Theta1, Theta2 = unpack_params(nn_params, input_layer_size, hidden_layer_size, num_labels)
    m = X.shape[0]

    # Forward propagation
    A_1 = X
    Z_2 = A_1.dot(Theta1.T)  # Ensure Theta1 has the correct size
    A_2 = add_zero_feature(sigmoid(Z_2))
    Z_3 = A_2.dot(Theta2.T)
    A_3 = sigmoid(Z_3)

    # Cost function
    J = -1/m * np.sum(Y * np.log(A_3) + (1 - Y) * np.log(1 - A_3))
    J += lambda_coef / (2 * m) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

    return J