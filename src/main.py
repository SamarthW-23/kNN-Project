from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from knn import KNN, accuracy
    
def main():
    iris = load_iris()
    X=iris.data
    y=iris.target

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    my_knn = KNN(k=5)
    my_knn.fit(X_train, y_train)
    y_pred_custom = my_knn.predict(X_test)
    custom_accuracy = accuracy(y_test, y_pred_custom)

    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test)
    sklearn_acc = accuracy_score(y_test, y_pred_sklearn)

    print("----- KNN Iris Classification -----")
    print(f"Custom KNN accuracy  : {custom_accuracy:.4f}")
    print(f"Sklearn KNN accuracy : {sklearn_acc:.4f}")

    def test_k_values(X_train, X_test, y_train, y_test, max_k=20):
        k_values = range(1, max_k + 1)
        accuracies = []

        for k in k_values:
            model = KNN(k=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy(y_test, y_pred)
            accuracies.append(acc)

        best_accuracy = max(accuracies)
        best_k = list(k_values)[accuracies.index(best_accuracy)]

        print(f"Best k: {best_k}")
        print(f"Best accuracy: {best_accuracy:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, accuracies, marker='o', linestyle='-')
        plt.xlabel("Value of k")
        plt.ylabel("Accuracy")
        plt.title("KNN Accuracy vs k")
        plt.xticks(range(1, max_k + 1))
        plt.grid(True)
        plt.show()
    
    test_k_values(X_train, X_test, y_train, y_test, max_k=20)


if __name__ == "__main__":
    main()