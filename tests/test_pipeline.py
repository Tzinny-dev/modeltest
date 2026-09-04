"""Tests for running scikit-learn Pipelines (feature engineering + model)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeltest import ModelSuite, TestContext
from modeltest.core.base import TestStatus
from modeltest.scenarios import (
    DataDriftTest,
    FeatureDominanceTest,
    MinimumAccuracyTest,
    TopFeaturesTest,
)
from modeltest.scenarios._utils import model_features


def _build_pipeline(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(45, 12, n),
            "income": rng.normal(60000, 20000, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    y = (df["income"] > df["age"] * 800 + rng.normal(0, 8000, n)).astype(int)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age", "income"]),
            ("cat", OneHotEncoder(), ["cat"]),
        ]
    )
    pipe = Pipeline(
        [("pre", pre), ("clf", RandomForestClassifier(n_estimators=30, random_state=0))]
    ).fit(df, y)
    return pipe, df, y


class TestPipelineSuite:
    def test_estimator_is_wrapped_as_sklearn(self):
        pipe, _, _ = _build_pipeline()
        from modeltest.wrappers import wrap

        assert isinstance(wrap(pipe).model, Pipeline)

    def test_suite_runs_pipeline(self):
        pipe, df, y = _build_pipeline()
        # Train a distinct frame to exercise drift detection, but keep drift small.
        df_train = df.copy()
        df_train["income"] = df_train["income"] + np.random.default_rng(1).normal(
            0, 5, len(df_train)
        )
        suite = ModelSuite(name="pipe")
        suite.add_tests(
            MinimumAccuracyTest(threshold=0.0),
            DataDriftTest(feature_cols=["age", "income"], max_psi=0.2),
        )
        result = suite.run(pipe, df, y, X_train=df_train, y_train=y)
        assert result.passed
        assert result.num_failed == 0

    def test_predict_with_extra_columns_filters_to_inputs(self):
        pipe, df, y = _build_pipeline()
        logging_col = df.copy()
        logging_col["_extra"] = np.random.default_rng(1).normal(size=len(df))
        X = model_features(pipe, logging_col)
        assert list(X.columns) == ["age", "income", "cat"]
        # prediction should succeed because X is filtered to the pipeline's inputs
        assert pipe.predict(X).shape[0] == len(df)


class TestPipelineExplainability:
    def test_attribution_uses_engineered_feature_names(self):
        pipe, df, y = _build_pipeline(n=600)
        ctx = TestContext(model=pipe, X_val=df, y_val=y)
        result = TopFeaturesTest(
            expected_features=["num__age", "num__income"], k=2
        ).run(ctx)
        assert result.status == TestStatus.PASSED

    def test_wrong_expected_engineered_features_fail(self):
        pipe, df, y = _build_pipeline(n=600)
        ctx = TestContext(model=pipe, X_val=df, y_val=y)
        result = TopFeaturesTest(expected_features=["cat__cat_a"], k=1).run(ctx)
        assert result.status == TestStatus.FAILED

    def test_dominance_on_pipeline(self):
        pipe, df, y = _build_pipeline(n=600)
        ctx = TestContext(model=pipe, X_val=df, y_val=y)
        result = FeatureDominanceTest(max_top_share=0.9).run(ctx)
        assert result.status == TestStatus.PASSED
