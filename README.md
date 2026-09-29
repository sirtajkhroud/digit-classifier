# Handwritten Digit Classifier

A small, self-contained machine-learning project that classifies handwritten
digits (0–9) from 8×8 grayscale images. It compares three classifiers, selects
the best by cross-validated accuracy, and evaluates it on a held-out test set.

Built with scikit-learn. Runs fully offline — the dataset ships with the library.

## Results

Trained on the scikit-learn `digits` dataset (1,797 images, 10 classes),
80/20 train/test split, 5-fold cross-validation:

| Model               | CV accuracy | Test accuracy | Macro-F1 |
|---------------------|:-----------:|:-------------:|:--------:|
| Logistic Regression |    0.970    |     0.972     |  0.972   |
| Random Forest       |    0.976    |     0.969     |  0.969   |
| **SVM (RBF)**       |  **0.983**  |   **0.981**   | **0.981**|

Best model: **RBF-kernel SVM — 98.1% test accuracy.**

## Quickstart

```bash
pip install -r requirements.txt
python train.py      # trains, evaluates, saves model.joblib
python predict.py    # loads the model and predicts on sample images
```

## Project structure

```
train.py          # loads data, compares models, evaluates, saves the best
predict.py        # loads the saved model and runs predictions
requirements.txt  # dependencies
```

## How it works

1. **Data** — 1,797 handwritten-digit images (8×8 = 64 pixel features each).
2. **Models** — logistic regression, random forest, and an RBF-kernel SVM,
   each wrapped in a scikit-learn `Pipeline` (with feature scaling where it helps).
3. **Selection** — 5-fold cross-validation on the training set picks the model;
   final numbers are reported on data the model never saw during training.
4. **Evaluation** — accuracy, macro-F1, and a full per-digit precision/recall report.

## Possible extensions

- Hyperparameter tuning with `GridSearchCV`.
- A confusion-matrix plot to see which digits get confused.
- Swap in the full MNIST dataset for a larger-scale version.
