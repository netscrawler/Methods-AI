import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from displayData import displayData

# Load data from .mat files
data = scipy.io.loadmat("test_set.mat")
weights = scipy.io.loadmat("weights.mat")

# Extract data
X = data["X"]
y = data["y"]
Theta1 = weights["Theta1"]
Theta2 = weights["Theta2"]

# Define parameter m
m = X.shape[0]  # Generate random indices
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]
# Display data
displayData(sel)

from predict import predict

# Get predictions
pred = predict(Theta1, Theta2, X)

# Evaluate accuracy
accuracy = np.mean(pred == y.ravel()) * 100
print(f"Accuracy: {accuracy}%")


rp = np.random.permutation(m)
plt.figure()
for i in range(5):
    X2 = X[rp[i], :]
    X2 = np.matrix(X[rp[i]])
    pred = predict(Theta1, Theta2, X2.getA())
    pred = np.squeeze(pred)
    pred_str = f"Neural Network Prediction: {pred} (digit {y[rp[i]]})"
    displayData(X2, pred_str)
plt.close()  # Get predictions
pred = predict(Theta1, Theta2, X)

# Determine error indices
incorrect_indices = np.where(pred != y.ravel())[0]

# Display first 100 errors
displayData(X[incorrect_indices[:100], :])
