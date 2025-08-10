from preprocess import load_and_split_data
from models import get_models
from train import train_model
from evaluate import evaluate_model
from visualize import (
    plot_confusion_matrix,
    plot_pairplot,
    plot_feature_importance,
    plot_decision_boundaries,
    plot_accuracy_comparison,
)
import numpy as np

if __name__ == "__main__":
    # Load and split data
    (X_train, X_test, y_train, y_test), target_names = load_and_split_data()

    # Use only 2 features for decision boundaries (first and second columns here)
    feature_idx = (0, 1)
    X_train_2f = X_train.iloc[:, list(feature_idx)]
    X_test_2f = X_test.iloc[:, list(feature_idx)]

    # Convert features to numpy arrays for plotting decision boundaries
    X_train_2f_np = X_train_2f.to_numpy()
    X_test_2f_np = X_test_2f.to_numpy()

    # y_train and y_test are already numpy arrays from label encoding, no to_numpy()
    y_train_np = y_train
    y_test_np = y_test

    # Plot pairplot once for entire dataset (loaded from sklearn for convenience)
    plot_pairplot()

    models = get_models()
    accuracy_summary = {}

    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")

        # Train and evaluate on 2 features only to keep decision boundaries consistent
        trained_model = train_model(model, X_train_2f, y_train)
        acc, report, cm = evaluate_model(
            trained_model, X_test_2f, y_test, target_names, model_name=model_name.replace(" ", "_")
        )

        plot_confusion_matrix(cm, labels=target_names, model_name=model_name.replace(" ", "_"))

        # Feature importance only for Logistic Regression
        if model_name == "Logistic Regression":
            plot_feature_importance(
                trained_model,
                feature_names=X_train_2f.columns,
                save_path=f"../results/feature_importance_{model_name.replace(' ', '_')}.png",
            )

        # Plot decision boundaries on 2 features
        plot_decision_boundaries(
            trained_model,
            X_train_2f_np,
            y_train_np,
            feature_idx=(0, 1),
            save_path=f"../results/decision_boundary_{model_name.replace(' ', '_')}.png",
        )

        accuracy_summary[model_name] = acc
        print(f"{model_name} Accuracy: {acc:.2f}\n")

    # Save accuracy summary text file
    with open("../results/summary_accuracy.txt", "w") as f:
        for name, acc in accuracy_summary.items():
            f.write(f"{name}: {acc:.4f}\n")

    # Plot accuracy comparison bar chart
    plot_accuracy_comparison(accuracy_summary)

    print("All models trained and evaluated. Check results/ folder for outputs.")
