from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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


if __name__ == "__main__":
    main()