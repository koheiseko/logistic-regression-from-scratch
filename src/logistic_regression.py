import numpy as np

class LogisticRegression_():
    def __init__(self, random_state=42):
        self._coef = list()
        self._intercept = int()
        self.random_state = np.random.seed(random_state)

    def _sigmoid(self, xi):
        return 1 / (1 + np.exp(-xi))

    def _cost_function(self, yi, pi):
        if yi == 1:
            return -np.log(pi)
        else:
            return -np.log(1 - pi)

    def _log_loss(self, y, p):
        m = y.shape[0]
    
        cost = sum([self._cost_function(pi, yi) for pi, yi in zip(p, y)])

        return 1 / m * cost 
    
    def _gradient(self, X, y, h):
        m = y.shape[0]
    
        return (X.T @ (h - y)) / m

    def fit(self, X, y, epochs=100, learning_rate=0.001):

        self._intercept, self._coef = np.array([0]), np.random.randn(X.shape[1])
        theta = np.concatenate((self._intercept, self._coef))
        
        X = np.c_[np.ones(X.shape[0]), X]
        z = X @ theta
        h = np.array([self._sigmoid(zi) for zi in z])

        for _ in range(epochs):
            gradients = self._gradient(X, y, h)
            
            theta = theta - learning_rate * gradients

            z = X @ theta
            h = np.array([self._sigmoid(zi) for zi in z])
        
        self._intercept, self._coef = np.array([theta[0]]), theta[1:]

        return self
        
    def predict(self, X, threshould=0.5):
        theta = np.concatenate((self._intercept, self._coef))

        X = np.c_[np.ones(X.shape[0]), X]

        z = X @ theta

        y_pred = (np.array([self._sigmoid(zi) for zi in z]) >= threshould).astype(int)

        return y_pred