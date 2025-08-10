# 🌸 Iris Flower Classification

This project uses the famous **Iris dataset** to classify flowers into three species:

* *Setosa*
* *Versicolor*
* *Virginica*

Based on four flower measurements:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

The goal is to build, train, and evaluate multiple machine learning models that predict the species from these measurements, and visualize their performance.

---

## 📂 Project Overview

* **Dataset**: [Iris CSV on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv?resource=download)
  The dataset includes an extra `Id` column which is ignored during training.

* **Algorithms**: Multiple supervised classification models including:

  * Logistic Regression
  * Decision Tree
  * k-Nearest Neighbors (k-NN)
  * Support Vector Machine (SVM)
  * Random Forest
  * Naive Bayes

* **Libraries Used**:

  * `scikit-learn` – dataset handling, preprocessing, model training and evaluation
  * `pandas` – data manipulation
  * `numpy` – numerical operations
  * `matplotlib` / `seaborn` – data and model visualization

---

## 🛠 Features

* Load and preprocess the Iris dataset from CSV, including label encoding
* Train multiple classification models and compare their accuracy
* Evaluate models with accuracy scores, confusion matrices, and classification reports
* Visualize model results:

  * Pairplot of the dataset features
  * Confusion matrices per model
  * Feature importance for Logistic Regression
  * Decision boundaries for all models (using two selected features)
  * Accuracy comparison bar chart

---

## ▶ Usage

Run the main script to train all models, evaluate, and generate visualizations:

```bash
python main.py
```

All outputs (reports, plots) will be saved in the `results/` folder.

---

## 📊 Evaluation

* **Metric Used**: Accuracy Score
* **Train/Test Split**: 80/20 by default
* Confusion matrix and classification report for each model provide detailed performance insight
* Accuracy comparison across models to identify the best performer

---

## 📈 Results Analysis

After training and evaluating multiple classification models on the Iris dataset, the following insights emerged:

* **Logistic Regression, SVM, and Naive Bayes** consistently achieved the highest accuracy (\~90%) using only two features. This suggests these models perform well with relatively simple decision boundaries on this dataset.

* **K-Nearest Neighbors (k-NN)** also showed strong performance (\~80%), benefiting from its instance-based learning approach, but may require tuning the number of neighbors (`k`) for optimal results.

* **Random Forest and Decision Tree** models performed moderately (\~63-77%), possibly due to training on only two features for visualization purposes. Their performance may improve when trained on all four features.

* The **confusion matrices** indicate that misclassifications mostly occur between *Versicolor* and *Virginica*, which are known to have overlapping feature distributions, whereas *Setosa* is generally easier to classify correctly.

* **Feature importance plots** for Logistic Regression reveal that petal length and petal width are more influential than sepal dimensions in distinguishing between species.

* **Decision boundary visualizations** clearly illustrate how different models partition the feature space, highlighting varying levels of model complexity and generalization.

---

### Suggestions for Future Improvement

* Training models on **all four features** instead of only two could potentially increase accuracy, especially for tree-based models.

* Implementing **hyperparameter tuning** (e.g., grid search) may optimize model performance.

* Using **cross-validation** rather than a single train-test split would provide more robust estimates of model generalization.
