import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_accuracy_vs_k(k_values, custom_accuracies, sklearn_accuracies=None):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, custom_accuracies, marker='o', label='Custom KNN')

    if sklearn_accuracies is not None:
        plt.plot(k_values, sklearn_accuracies, marker='s', linestyle='--', label='Sklearn KNN')

    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs k")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_test_accuracy(k_values, train_accuracies, test_accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(k_values, test_accuracies, marker='s', label='Testing Accuracy')

    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_meshgrid(X, step=0.02, padding=1.0):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    return xx, yy


def plot_decision_regions(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    title="KNN Decision Regions",
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    xx, yy = create_meshgrid(X_train)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = np.array(Z).reshape(xx.shape)

    cmap_light = ListedColormap(["#FFCCCC", "#CCFFCC", "#CCCCFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00AA00", "#0000FF"])

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_light)

    train_scatter = ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train, cmap=cmap_bold,
        marker='o', edgecolor='k', s=60, label='Train'
    )

    ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap=cmap_bold,
        marker='x', s=90, linewidths=2, label='Test'
    )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    ax.grid(True)

    return train_scatter

def plot_feature_pair_grid(results, overall_title="KNN Decision Regions for All Feature Pairs"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    scatter_for_legend = None

    for i, result in enumerate(results):
        feature_names = result["feature_names"]
        accuracy = result["accuracy"]
        model = result["model"]
        X_train = result["X_train"]
        X_test = result["X_test"]
        y_train = result["y_train"]
        y_test = result["y_test"]

        scatter_for_legend = plot_decision_regions(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            title=f"{feature_names[0]} vs {feature_names[1]}\nAccuracy = {accuracy:.3f}",
            ax=axes[i]
        )

    handles, _ = scatter_for_legend.legend_elements()
    class_labels = ["Setosa", "Versicolor", "Virginica"]

    fig.legend(
        handles[:3], class_labels,
        loc="lower center",
        ncol=3,
        fontsize=11
    )

    fig.suptitle(overall_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    def plot_single_feature_pair_result(result):
        plot_decision_regions(
            model=result["model"],
            X_train=result["X_train"],
            X_test=result["X_test"],
            y_train=result["y_train"],
            y_test=result["y_test"],
            feature_names=result["feature_names"],
            title=f"{result['feature_names'][0]} vs {result['feature_names'][1]} | Accuracy = {result['accuracy']:.3f}"
        )
        plt.legend(["Train", "Test"])
        plt.tight_layout()
        plt.show()