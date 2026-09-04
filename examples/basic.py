"""End-to-end example: define a suite and run it against a model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from modeltest import ModelSuite
from modeltest.scenarios import (
    DataDriftTest,
    EqualOpportunityTest,
    GroupPerformanceTest,
    MinimumAccuracyTest,
    RobustnessTest,
    StatisticalParityTest,
)

rng = np.random.default_rng(0)
n = 2000
gender = rng.choice(["M", "F"], n)
female = (gender == "F").astype(int)
X = pd.DataFrame(
    {
        "age": rng.normal(45, 12, n),
        "income": rng.normal(60000, 20000, n),
        "gender_male": (gender == "M").astype(int),
        "_gender": gender,  # keep original for subgroup grouping
    }
)
y = (X["income"] > X["age"] * 800 + rng.normal(0, 8000, n)).astype(int)

FEATURES = ["age", "income", "gender_male"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(n_estimators=50, random_state=0).fit(
    X_train[FEATURES], y_train
)

suite = ModelSuite(name="Income Model")
suite.add_tests(
    MinimumAccuracyTest(threshold=0.85),
    GroupPerformanceTest(metric="accuracy", threshold=0.8, group_col="_gender"),
    RobustnessTest(noise_std=10, max_drop=0.05, metric="accuracy"),
    DataDriftTest(feature_cols=["age", "income"], max_psi=0.1),
    EqualOpportunityTest(protected_col="_gender", max_diff=0.1),
    StatisticalParityTest(protected_col="_gender", max_diff=0.1, min_ratio=0.8),
)

result = suite.run(
    model,
    X_val[FEATURES + ["_gender"]],
    y_val,
    X_train=X_train[FEATURES + ["_gender"]],
    model_name="income_rf",
)
print(result.report(style="table"))
