import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris

def plot_pairplot(save_path="../results/pairplot.png"):
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df['species'] = df['target'].apply(lambda i: iris.target_names[i])

    sns.pairplot(df, hue="species", diag_kind="kde", palette="Set2")
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path="../results/feature_importance.png"):
    if not hasattr(model, 'coef_'):
        print("Model has no coef_ attribute, skipping feature importance plot.")
        return
    importance = np.mean(np.abs(model.coef_), axis=0)
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45)
    plt.ylabel("Importance (avg |coef|)")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_decision_boundaries(model, X, y, feature_idx=(0, 1), save_path="../results/decision_boundary.png"):
    X_sub = X[:, feature_idx]
    x_min, x_max = X_sub[:, 0].min() - 0.5, X_sub[:, 0].max() + 0.5
    y_min, y_max = X_sub[:, 1].min() - 0.5, X_sub[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    except Exception as e:
        print(f"Failed to predict decision boundaries: {e}")
        return

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set2)
    plt.scatter(X_sub[:, 0], X_sub[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set2)
    iris = load_iris()
    plt.xlabel(iris.feature_names[feature_idx[0]])
    plt.ylabel(iris.feature_names[feature_idx[1]])
    plt.title("Decision Boundaries")
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, labels, model_name="model", save_path_prefix="../results/confusion_matrix_"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"{save_path_prefix}{model_name}.png")
    plt.close()


def plot_accuracy_comparison(accuracy_dict, save_path="../results/accuracy_comparison.png"):
    names = list(accuracy_dict.keys())
    scores = list(accuracy_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, scores, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45, ha='right')

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05,
                 f"{score:.2f}", ha='center', color='black', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
