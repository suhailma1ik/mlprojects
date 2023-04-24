import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent optimization
        for iteration in range(self.num_iterations):
            # Compute the margin and the hinge loss
            margin = y * (X.dot(self.weights) + self.bias)
            hinge_loss = np.maximum(0, 1 - margin)
            
            # Compute the gradients of the weights and the bias
            d_weights = self.lambda_param * self.weights - np.mean(X * y[:, np.newaxis] * (margin < 1), axis=0)
            d_bias = -np.mean(y * (margin < 1))
            
            # Update the parameters using the gradients
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
    
    def predict(self, X):
        # Predict the class labels of the input data
        margin = X.dot(self.weights) + self.bias
        return np.sign(margin)
