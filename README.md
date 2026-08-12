# Machine Learning Assignment 2 — Breast Cancer Classification Dashboard

## a) Problem Statement
The goal of this project is to build, evaluate, and deploy multiple binary
classification models that predict whether a breast tumor is **malignant**
or **benign** based on measurements extracted from digitized images of a
fine needle aspirate (FNA) of a breast mass. Early and accurate
classification of tumors directly supports faster, more reliable clinical
diagnosis. Five classification algorithms are trained on the same dataset,
evaluated using six standard metrics, and made available for interactive
comparison through a Streamlit web application.

## b) Dataset Description
- **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset — a well-known
  public dataset (available directly via `sklearn.datasets.load_breast_cancer`,
  originally from UCI Machine Learning Repository).
- **Instances:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst"
  values of 10 real-valued measurements per cell nucleus: radius, texture,
  perimeter, area, smoothness, compactness, concavity, concave points,
  symmetry, fractal dimension)
- **Target:** Binary — 0 = Malignant, 1 = Benign
- **Class balance:** 212 malignant / 357 benign
- **Split:** 80% train / 20% test, stratified by class (random_state=42)
- Features were standardized (z-score scaling) for the scale-sensitive
  models (Logistic Regression, kNN).

## c) GitHub Repository Link
`https://github.com/deba79kundu-droid/ML_Assignment_2`

## d) Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset. The classes are close to linearly separable after scaling, so a linear decision boundary works very well. High precision and recall mean very few malignant cases are missed and very few benign cases are misclassified as malignant. |
| Decision Tree | Weakest of the five. A single tree (max_depth=5) overfits certain splits and is more sensitive to noise in individual features, giving the lowest AUC and MCC of the group. |
| kNN | Very strong — highest recall (1.00), meaning it caught every malignant case in the test set, which matters a lot in a medical screening context. Distance-based method benefits directly from feature scaling. |
| Naive Bayes | Solid AUC despite the "naive" independence assumption between features being violated here (many of the 30 features are correlated, e.g., radius/perimeter/area). Accuracy and MCC are moderate compared to the top models. |
| Random Forest (Ensemble) | Very consistent — high AUC and balanced precision/recall, since averaging many trees reduces the overfitting seen in the single Decision Tree. Slightly behind Logistic Regression and kNN on this particular dataset, but generally more robust to new/unseen data distributions. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy, AUC, Precision, Recall, F1, and MCC. **kNN** is the runner-up and arguably the safer clinical choice given its perfect recall (no missed malignant cases). |

## App Features
- CSV upload for test data
- Model selection dropdown (5 models)
- Live evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap + full classification report

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # trains models, generates test_data.csv + metrics.csv
streamlit run app.py
```

## Repository Structure
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
│   │-- train_models.py
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- scaler.pkl
│   │-- metrics.csv
│   │-- metrics.json
```
