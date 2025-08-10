from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, target_names, model_name="model"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    with open(f"../results/classification_report_{model_name}.txt", "w") as f:
        f.write(f"Accuracy: {acc:.2f}\n\n{report}")

    return acc, report, cm
