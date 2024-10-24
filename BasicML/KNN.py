import numpy as np
from collections import Counter

class KNN:
    def __init__(self, X, y,k=3):
        self.k = k
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Step 1: Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Step 2: Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Step 3: Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Step 4: Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def main():
    # Example dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # n points 
    y = np.array([0, 0, 1, 1, 1])  # labels

    # Test point
    X_test = np.array([[1.5, 2.5]])

    # Create a KNN instance with k=3
    knn = KNN(X, y, k=3)
    #knn.fit(X, y)

    # Predict the class label for the test point
    prediction = knn.predict(X_test)

    print(f"Predicted class for {X_test[0]}: {prediction[0]}")


# Ensure this block runs only if the script is run directly
if __name__ == "__main__":
    main()  # Call the main function