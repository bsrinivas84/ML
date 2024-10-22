import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=10000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def costFunction(self, y_pred, y_actual):
        cost = np.mean((y_pred - y_actual) ** 2)
        return cost

    def gradientDescent(self, W, Y, X):
        n_samples, n_features = X.shape
        previous_cost = float('inf')  # Initialize previous cost to infinity
        for i in range(self.n_iters):
            y_pred = np.dot(X, W) + self.bias
            cost = self.costFunction(y_pred, Y)
            dW = (1/n_samples) * np.dot(X.T, (y_pred - Y))
            db = (1/n_samples) * np.sum(y_pred - Y)
            
            # Update weights and bias
            W = W - self.learning_rate * dW
            self.bias = self.bias - self.learning_rate * db

            # Append cost to history
            self.cost_history.append(cost)

            # Early stopping condition
            if abs(previous_cost - cost) < self.tolerance:
                print(f"Early stopping at iteration {i}, cost: {cost}")
                break

            previous_cost = cost

            # Optionally print every 100 iterations
            if i % 100 == 0:
                print(f"Cost at iteration {i} is {cost}")

        return W, self.bias

    def fit(self, X, Y):
        # Standardize the features (mean = 0, std = 1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.weights, self.bias = self.gradientDescent(self.weights, Y, X)

    def predict(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def plot_cost(self):
        plt.plot(self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost (MSE)")
        plt.title("Cost over iterations")
        plt.show()

# Sample data
X = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]).T
Y = np.array([3, 4, 5, 6, 7])

print("Entered main")
lr = LinearRegression()
print("After LinearRegression")
lr.fit(X, Y)
print(lr.predict(X))
lr.plot_cost()  # Plot the cost function over iterations
