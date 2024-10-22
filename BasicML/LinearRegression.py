import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=10000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    


    def costFunction(self, y_pred, y_actual):
       cost = np.mean((y_pred - y_actual )**2)
       return cost
    

    def gradientDescent(self,W, Y, X):
        n_samples, n_features = X.shape
        for i in range(self.n_iters):
            y_pred = np.dot(X,W) + self.bias
            cost = self.costFunction(y_pred, Y)
            dW = (1/n_samples) * np.dot(X.T, (y_pred - Y))       
            db = (1/n_samples) * np.sum(y_pred - Y)
            self.weights = self.weights - self.learning_rate * dW
            self.bias = self.bias - self.learning_rate * db

            cost = self.costFunction(y_pred, Y)
            print(cost)
            if i % 100 == 0:
                print(f"Cost at iteration {i} is {cost}")

        return self.weights, self.bias  


    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.weights, self.bias = self.gradientDescent(self.weights, Y, X)

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred   
    
X = np.array([[1,2,3,4,5], [2,3,4,5,6]]).T
Y = np.array([3,4,5,6,7])
print("entered main")
lr = LinearRegression()
print("After LinearRegression")
lr.fit(X,Y)
print(lr.predict(X))
    
