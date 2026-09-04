"""Generate sample model + data for the CLI/YAML demo."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(__file__))
rng = np.random.default_rng(0)
n = 3000
gender = rng.choice(["M", "F"], n)
X = pd.DataFrame(
    {
        "age": rng.normal(45, 12, n),
        "income": rng.normal(60000, 20000, n),
        "gender_male": (gender == "M").astype(int),
        "gender": gender,
    }
)
y = (X["income"] > X["age"] * 800 + rng.normal(0, 16000, n)).astype(int)
df = X.copy()
df["target"] = y

FEATURES = ["age", "income", "gender_male"]

X_train, X_val = train_test_split(df, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=60, random_state=0).fit(
    X_train[FEATURES], X_train["target"]
)

joblib.dump(model, os.path.join(OUT, "model.pkl"))
X_train.to_csv(os.path.join(OUT, "train.csv"), index=False)
X_val.to_csv(os.path.join(OUT, "validation.csv"), index=False)
print("wrote model.pkl, train.csv, validation.csv")
