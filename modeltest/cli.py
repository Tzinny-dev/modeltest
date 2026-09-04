"""Command-line interface: `modeltest validate`."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from modeltest.core.base import ModelSuite


def _build_pass() -> bool:
    return True


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modeltest", description="Unit tests for machine learning models."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Run a test suite against a model.")
    validate.add_argument(
        "--suite",
        required=True,
        help="Path to suite.py (defining `suite`) or suite.yaml (declarative).",
    )
    validate.add_argument("--model", required=True, help="Path to model (pickle).")
    validate.add_argument("--data", required=True, help="Path to validation CSV.")
    validate.add_argument(
        "--target", default="target", help="Name of the target column."
    )
    validate.add_argument(
        "--output", default=None, help="Write JUnit XML to this path."
    )
    validate.add_argument(
        "--train-data", default=None, help="Optional training CSV (for drift tests)."
    )
    validate.set_defaults(func=_run_validate)

    args = parser.parse_args(argv)
    return args.func(args)


def _load_suite(path: str) -> "ModelSuite":
    import os

    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        from modeltest.config import load_suite_yaml

        return load_suite_yaml(path)
    # default: treat as a Python module exposing `suite`
    import importlib.util

    spec = importlib.util.spec_from_file_location("suite_module", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.suite


def _run_validate(args: argparse.Namespace) -> int:
    import os

    import joblib
    import pandas as pd

    suite: ModelSuite = _load_suite(args.suite)
    model = joblib.load(os.path.abspath(args.model))
    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    X_train = None
    if args.train_data:
        train_df = pd.read_csv(args.train_data)
        X_train = train_df.drop(columns=[args.target])

    result = suite.run(
        model, X, y, X_train=X_train, model_name=os.path.basename(args.model)
    )

    if args.output:
        from modeltest.core.report import to_junit_xml

        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(to_junit_xml(result))

    print(result.report(style="table"))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
