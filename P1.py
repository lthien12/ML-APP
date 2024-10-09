import numpy as np
import matplotlib.pyplot as plt

# Initial Values
x = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210], dtype='float64')
y = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80], dtype='float64')
theta0 = 0
theta1 = 0
learning_rate = 1e-5
epochs = 1000
epsilon = 1e-8

# Cost Function
def cost(theta0, theta1):
    m = x.shape[0]
    return 1/(2*m) * np.sum((predict(theta0, theta1) - y)**2) 

# Calculate y_hat
def predict(theta0, theta1):
    return theta0 + theta1*x

# Gradient Descent
def gradient_descent(theta0, theta1):
    m = x.shape[0]
    theta0 = theta0 - learning_rate * 1/m * np.sum((predict(theta0, theta1) - y)) 
    theta1 = theta1 - learning_rate * 1/m * np.sum((predict(theta0, theta1) - y) * x) 
    return theta0, theta1

if __name__ == '__main__':
    delta_j = np.Infinity
    for i in range(epochs):
        if abs(delta_j) > epsilon:
            j = cost(theta0, theta1)
            theta0, theta1 = gradient_descent(theta0, theta1)
            j_new = cost(theta0, theta1)
            delta_j = j_new - j
            j = j_new
            print(f'iterators: {i+1}, cost: {j}')
        else:
            break

    print(f'\ntheta0: {theta0}, theta1: {theta1}')

    # Display the plot
    fig, ax = plt.subplots()
    ax.set(xlabel='Study time(h)', ylabel='Exam score')
    ax.axis([0, 500, 0, 100])
    ax.scatter(x, y, s=100, facecolor='navy')
    x = np.arange(0, 500, 1)
    y_hat = predict(theta0, theta1)
    ax.plot(x, y_hat)
    plt.show()
