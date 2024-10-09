import numpy as np
import csv

# Feature Scaling
def feature_scaling(X):
    mean_val = np.mean(X, axis=0)
    max_val = np.max(X, axis=0)
    X = (X - mean_val) / max_val
    return X, mean_val, max_val

# Rescaling parameters
def inverse_scaling(theta, mean_val, max_val):
    max_val = max_val.reshape(-1, 1)
    mean_val = mean_val.reshape(-1, 1)
    theta_original = np.zeros_like(theta)
    theta_original[0] = theta[0] - np.sum((theta[1:] * mean_val) / max_val)
    theta_original[1:] = theta[1:] / max_val

    return theta_original

# Cost Function
def cost(X, y, theta):
    m = X.shape[0]
    return 1 / (2*m) * np.sum((predict(X, theta) - y) ** 2)

# Calculate y_hat
def predict(X, theta):
    return X@theta

# Gradient Descent
def gradient_descent(X, y, theta):
    m = X.shape[0]
    theta -=  learning_rate / m * (X.T) @ ((predict(X, theta) - y))
    return theta

if __name__ == '__main__':
    #Read data from csv file
    with open('Practice2_Chapter2.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        data = np.array([row for row in csv_reader], dtype='float64')

    # Initial Values
    X = data[:, :3]
    X, mean_val, max_val = feature_scaling(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    y = data[:, -1].reshape(-1, 1)
    theta = np.zeros(X.shape[1], dtype='float64').reshape(-1, 1)

    learning_rate = 0.5
    epochs = 1000
    epsilon = 1e-15
    delta_j = np.Infinity

    # Training loop
    for i in range(epochs):
        if abs(delta_j) > epsilon:
            j = cost(X, y, theta)
            theta = gradient_descent(X, y, theta)
            j_new = cost(X, y, theta)
            delta_j = j_new - j
            j = j_new
            print(f'iterators: {i+1}, cost: {j}')
        else:
            break

    theta = inverse_scaling(theta, mean_val, max_val)
    print(f'\ntheta: {theta.reshape(-1)}')

    # Predict
    tv_value = 57.5
    radio_value = 32.8
    newspaper_value = 23.5
    x = np.array([1, tv_value, radio_value, newspaper_value], dtype='float64').reshape(1, -1)
    print(f'\nThe predicted sales for TV({tv_value}), Radio({radio_value}) and Newspaper({newspaper_value}): {predict(x, theta)[0, 0]}')
