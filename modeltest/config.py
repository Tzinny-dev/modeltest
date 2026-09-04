"""Declarative suite definition: load a ModelSuite from YAML.

The YAML format mirrors the built-in scenarios with a ``type`` and ``params``:

.. code-block:: yaml

    suite:
      name: "Credit Scoring Model"
      tests:
        - type: minimum_accuracy
          params:
            threshold: 0.85
        - type: group_performance
          params:
            metric: accuracy
            threshold: 0.8
            group_col: "gender"
        - type: robustness
          params: {noise_std: 0.01, max_drop: 0.03}
        - type: data_drift
          params: {features: [age, income], max_psi: 0.15}
        - type: equal_opportunity
          params: {protected_col: "gender", max_diff: 0.1}
        - type: statistical_parity
          params: {protected_col: "gender", max_diff: 0.1, min_ratio: 0.8}
        - type: feature_dominance
          params: {max_top_share: 0.9}
        - type: top_features
          params: {expected_features: [income, age], k: 2}
        - type: confidence_threshold
          params: {metric: accuracy, threshold: 0.85, n_boot: 1000, alpha: 0.05}
        - type: data_invariant
          params: {expected_columns: [age, income], max_null_ratio: 0.02}
"""

from __future__ import annotations

import importlib
import os

import yaml

from modeltest.core.base import ModelSuite, ModelTest
from modeltest.scenarios import (  # type: ignore[attr-defined]
    ConfidenceThresholdTest,
    DataDriftTest,
    DataInvariantTest,
    EqualOpportunityTest,
    FeatureDominanceTest,
    GroupPerformanceTest,
    KSTest,
    MinimumAccuracyTest,
    NoNullTest,
    RobustnessTest,
    StatisticalParityTest,
    TopFeaturesTest,
)

# Map YAML `type` strings to the scenario class used to build each test.
_REGISTRY = {
    "minimum_accuracy": MinimumAccuracyTest,
    "group_performance": GroupPerformanceTest,
    "robustness": RobustnessTest,
    "data_invariant": DataInvariantTest,
    "no_null": NoNullTest,
    "data_drift": DataDriftTest,
    "ks": KSTest,
    "equal_opportunity": EqualOpportunityTest,
    "statistical_parity": StatisticalParityTest,
    "feature_dominance": FeatureDominanceTest,
    "top_features": TopFeaturesTest,
    "confidence_threshold": ConfidenceThresholdTest,
}


def register(type_name: str, cls: type) -> None:
    """Map a YAML ``type`` string to a custom test class.

    Custom tests must subclass :class:`modeltest.ModelTest`. Registering
    before :func:`load_suite_yaml` lets a suite reference your test by name::

        from modeltest.config import register
        register("my_error_check", MyErrorCheck)

        # suite.yaml
        # suite:
        #   tests:
        #     - type: my_error_check
        #       params: {max_errors: 5}
    """
    if not (isinstance(cls, type) and issubclass(cls, ModelTest)):
        raise TypeError(f"register() expects a ModelTest subclass, got {cls!r}")
    _REGISTRY[type_name] = cls


def unregister(type_name: str) -> None:
    """Remove a previously registered type (built-ins included)."""
    _REGISTRY.pop(type_name, None)


def _resolve_type(type_name: str) -> type:
    """Look up a test class by registered name or a dotted import path.

    Unknown names may be an import reference of the form ``module.path:Class``
    or ``module.path.Class`` pointing at a custom test class. This lets a YAML
    suite load tests defined outside modeltest without a ``register`` call.
    """
    if type_name in _REGISTRY:
        return _REGISTRY[type_name]

    if ":" in type_name:
        module_path, _, attr = type_name.partition(":")
        cls = _import_attr(module_path, attr)
    else:
        module_path, _, attr = type_name.rpartition(".")
        cls = _import_attr(module_path, attr) if module_path else None

    if cls is None:
        raise ValueError(
            f"Unknown test type {type_name!r}. Known: "
            f"{', '.join(sorted(_REGISTRY))}. Or use module.path:Class."
        )
    return cls


def _import_attr(module_path: str, attr: str) -> type:
    module = _try_import(module_path)
    cls = getattr(module, attr, None)
    if cls is None:
        raise ValueError(f"Module {module_path!r} has no attribute {attr!r}")
    if not (isinstance(cls, type) and issubclass(cls, ModelTest)):
        raise TypeError(f"{module_path}.{attr} is not a ModelTest subclass")
    return cls


def _try_import(module_path: str):
    import sys

    try:
        return importlib.import_module(module_path)
    except ImportError:
        # User-defined test modules are often plain files in the working
        # directory; make sure CWD is importable so dotted paths "just work"
        # from the CLI as well as from library code.
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            return importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(f"Could not import module {module_path!r}: {exc}") from exc


def load_suite_yaml(path: str) -> ModelSuite:
    """Build a :class:`ModelSuite` from a YAML file.

    Raises ``ValueError`` for unknown test types or malformed structure.
    """
    with open(path) as fh:
        doc = yaml.safe_load(fh)

    suite_cfg = doc.get("suite")
    if not isinstance(suite_cfg, dict):
        raise ValueError("YAML must contain a top-level `suite:` mapping")

    name = suite_cfg.get("name", "suite")
    suite = ModelSuite(name=name)

    for raw in suite_cfg.get("tests", []):
        suite.add_test(_build_test(raw))

    return suite


def _build_test(raw) -> object:
    if not isinstance(raw, dict):
        raise ValueError(f"Each test must be a mapping, got: {raw!r}")

    type_name = raw.get("type")
    cls = _resolve_type(type_name)

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"`params` for {type_name!r} must be a mapping")

    # Accept clear YAML-friendly aliases for a couple of constructor names.
    params = {_PARAM_ALIASES.get(k, k): v for k, v in params.items()}

    # Ignore unknown params instead of crashing, but surface typos for hard cases.
    import inspect

    sig = inspect.signature(cls.__init__)
    valid = {k for k in sig.parameters if k not in ("self", "args", "kwargs")}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(
            f"Unknown params for {type_name!r}: {sorted(unknown)}. "
            f"Valid: {sorted(valid)}"
        )
    return cls(**params)


_PARAM_ALIASES = {
    "features": "feature_cols",
    "columns": "feature_cols",
    "group": "group_col",
    "protected": "protected_col",
}


def dump_suite_yaml(suite: ModelSuite, path: str) -> None:
    """Serialise a suite back to YAML (best-effort; only built-in tests)."""
    import inspect

    doc = {"suite": {"name": suite.name, "tests": []}}
    for test in suite.tests:
        params = {
            k: v
            for k, v in vars(test).items()
            if not k.startswith("_")
            and k in inspect.signature(type(test).__init__).parameters
        }
        doc["suite"]["tests"].append({"type": _inverse_name(test), "params": params})
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def _inverse_name(test) -> str:
    inverse = {v: k for k, v in _REGISTRY.items()}
    return inverse.get(type(test), type(test).__name__)
