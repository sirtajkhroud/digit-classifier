"""
Handwritten digit classifier.
Compares three models on the scikit-learn `digits` dataset (1,797 8x8 images,
10 classes), evaluates the best on a held-out test set, and saves it.
Runs fully offline. Usage: python train.py
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

RANDOM_STATE = 42

def main():
    X, y = load_digits(return_X_y=True)
    print(f"Dataset: {X.shape[0]} images, {X.shape[1]} features (8x8 pixels), {len(set(y))} classes\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10, gamma="scale")),
        ]),
    }

    results = []
    for name, pipe in candidates.items():
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"{name:>18} | CV acc {cv.mean():.3f} (+/-{cv.std():.3f}) | test acc {acc:.3f} | macro-F1 {f1:.3f}")
        results.append({"name": name, "acc": acc, "f1": f1, "model": pipe})

    best = max(results, key=lambda r: r["acc"])
    print(f"\nBest model: {best['name']} — test accuracy {best['acc']:.3f}, macro-F1 {best['f1']:.3f}\n")

    preds = best["model"].predict(X_test)
    print("Per-class performance (best model):")
    print(classification_report(y_test, preds, digits=3))

    joblib.dump(best["model"], "model.joblib")
    print("Saved best model -> model.joblib")

if __name__ == "__main__":
    main()
