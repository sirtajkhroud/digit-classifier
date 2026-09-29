"""
Load the trained model and predict on sample images from the digits dataset.
Usage: python predict.py
"""
import numpy as np
from sklearn.datasets import load_digits
import joblib

def main():
    model = joblib.load("model.joblib")
    X, y = load_digits(return_X_y=True)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=10, replace=False)
    preds = model.predict(X[idx])

    print("sample | predicted | actual")
    for i, p, a in zip(idx, preds, y[idx]):
        flag = "" if p == a else "  <-- miss"
        print(f"{i:>6} | {p:^9} | {a:^6}{flag}")

if __name__ == "__main__":
    main()
