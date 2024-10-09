import numpy as np

# Data
X = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 
              0.364, 0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 
              0.561, 0.569, 0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 
              1.036, 1.045])
y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
              1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the bias term

# Logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(y, y_pred):
    return -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

# Gradient Descent
def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    
    for epoch in range(epochs):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        
        # Calculate the gradient
        gradient = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradient
        
        # Optionally print the cost every 100 epochs
        if epoch % 100 == 0:
            cost = cost_function(y, y_pred)
            print(f'Epoch {epoch}, Cost: {cost}')
    
    return weights

# Train the model
weights = logistic_regression(X, y)

# Predictions
def predict(X, weights):
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)

# Make predictions on the training set
predictions = predict(X, weights)

# Print weights and predictions
print('Weights:', weights)
print('Predictions:', predictions)
