# Heart Disease Prediction using Ensemble Learning

A comparative study of classical machine learning models — Decision Tree, AdaBoost, Gradient Boosting, and a Voting Ensemble — for predicting the presence of heart disease from clinical attributes.

## Overview

Cardiovascular disease is one of the leading causes of death worldwide. Early risk prediction from routine clinical measurements can support timely medical intervention. This project builds and evaluates multiple classifiers on the UCI Heart Disease dataset, with a focus on ensemble methods (bagging, boosting, and soft voting) to improve predictive reliability over a single baseline model.

## Dataset

- **Source:** UCI Heart Disease Dataset (Cleveland subset)
- **Samples:** 303 patient records
- **Features:** 13 clinical attributes — age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- **Target:** Binary classification — presence (1) or absence (0) of heart disease
- **Split:** 80/20 train-test, stratified

## Models

| Notebook | Model | Description |
|---|---|---|
| `decision_tree_heart.ipynb` | Decision Tree | Baseline classifier (max depth = 5) |
| `adaboost_heart.ipynb` | AdaBoost | Boosting ensemble of depth-1 decision stumps (100 estimators) |
| `gradient_boosting_heart.ipynb` | Gradient Boosting | Sequential error-correction boosting with hyperparameter tuning |
| `ensemble_voting_classifier_heart.ipynb` | Voting Ensemble | Soft-voting combination of Logistic Regression, Decision Tree, and SVM |

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 78.69% | 0.7500 | 0.9091 | 0.8219 |
| AdaBoost | 81.97% | 0.7895 | 0.9091 | 0.8451 |
| Gradient Boosting | 81.97% | 0.7895 | 0.9091 | 0.8451 |
| Voting Ensemble (LR + DT + SVM) | 80.33% | 0.7838 | 0.8788 | 0.8286 |
| Random Forest | 83.61% | 0.7805 | 0.9697 | 0.8649 |

## Key Findings

- Boosting methods (AdaBoost, Gradient Boosting) improved accuracy by ~3.3 percentage points over the single Decision Tree baseline, confirming that sequential error correction reduces bias.
- All ensemble models achieved recall above 0.87 — critical in medical diagnosis, where missing a true positive case carries a higher cost than a false alarm.
- Random Forest delivered the best overall performance, showing that bagging captures feature interactions in this dataset more effectively than a heterogeneous voting blend.

## Tech Stack

- Python
- scikit-learn
- pandas / NumPy
- Matplotlib
- Jupyter Notebook

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook
```

Open any notebook and run all cells. Each notebook is self-contained and uses `heart.csv` from the same directory.

## Project Structure

```
.
├── heart.csv
├── decision_tree_heart.ipynb
├── adaboost_heart.ipynb
├── gradient_boosting_heart.ipynb
├── ensemble_voting_classifier_heart.ipynb
└── README.md
```
