import numpy as np
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.knn import KNN, accuracy

def load_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names

def select_features(X, feature_indices):
    return X[:, feature_indices]

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def run_custom_knn(X_train, X_test, y_train, y_test, k=5):
    model = KNN(k=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "accuracy": acc
    }

def run_sklearn_knn(X_train, X_test, y_train, y_test, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "accuracy": acc
    }

def evaluate_k_range(X_train, X_test, y_train, y_test, k_min=1, k_max=20):
    k_values = list(range(k_min, k_max + 1))
    custom_accuracies = []
    sklearn_accuracies = []

    for k in k_values:
        custom_result = run_custom_knn(X_train, X_test, y_train, y_test, k=k)
        sklearn_result = run_sklearn_knn(X_train, X_test, y_train, y_test, k=k)

        custom_accuracies.append(custom_result["accuracy"])
        sklearn_accuracies.append(sklearn_result["accuracy"])

    return {
        "k_values": k_values,
        "custom_accuracies": custom_accuracies,
        "sklearn_accuracies": sklearn_accuracies
    }

def get_feature_pairs(n_features):
    return list(combinations(range(n_features), 2))

def run_feature_pair_experiment(feature_pair, k=5, test_size=0.2, random_state=42):
    X, y, feature_names, target_names = load_iris_data()
    X_pair = select_features(X, feature_pair)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X_pair, y, test_size=test_size, random_state=random_state
    )

    custom_result = run_custom_knn(X_train, X_test, y_train, y_test, k=k)

    return {
        "feature_pair": feature_pair,
        "feature_names": [feature_names[feature_pair[0]], feature_names[feature_pair[1]]],
        "target_names": target_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": custom_result["model"],
        "predictions": custom_result["predictions"],
        "accuracy": custom_result["accuracy"],
        "scaler": scaler
    }


def run_all_feature_pair_experiments(k=5):
    X, y, feature_names, target_names = load_iris_data()
    pairs = get_feature_pairs(X.shape[1])

    results = []
    for pair in pairs:
        result = run_feature_pair_experiment(pair, k=k)
        results.append(result)

    return results


#load_iris_data()
#split_and_scale(X, y)
#evaluate_k_range(X_train, X_test, y_train, y_test, max_k=20)
#get_feature_pair_data(feature_idx1, feature_idx2)