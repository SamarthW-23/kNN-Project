import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict_one(self, x_test):
        distances = []

        for x_train in self.X_train:
            dist = self.euclidean_distance(x_test, x_train)
            distances.append(dist)

        distances = np.array(distances)
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]

        prediction = Counter(nearest_labels).most_common(1)[0][0]
        return prediction

    def predict(self, X_test):
        predictions = []

        for x_test in X_test:
            prediction = self.predict_one(x_test)
            predictions.append(prediction)

        return np.array(predictions)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
