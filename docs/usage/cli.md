# CLI

`modeltest` ships one command: `validate`. It loads a suite, a pickled model
and validation data, runs every test, prints a table and exits non-zero if
any test fails.

```bash
# Declarative suite (recommended)
modeltest validate \
  --suite suite.yaml \
  --model model.pkl \
  --data validation.csv \
  --target target \
  --train-data train.csv \
  --output reports/model-validation.xml

# Python suite (a suite.py file exposing a `suite` object)
modeltest validate --suite suite.py --model model.pkl --data validation.csv --target target
```

## Flags

| Flag | Required | Default | Description |
| ---- | -------- | ------- | ----------- |
| `--suite` | yes | — | Path to a `suite.yaml`/`suite.yml` file, or to a Python file exposing a `suite` object. |
| `--model` | yes | — | Path to the pickled model (loaded with `joblib.load`). |
| `--data` | yes | — | Path to the validation CSV. |
| `--target` | no | `target` | Name of the target column; it is split off `X`. |
| `--output` | no | none | Write the JUnit XML report to this path (created if needed). |
| `--train-data` | no | none | Training CSV — required by the [drift tests](../scenarios/drift.md). |

## Behavior

- The **table report** is always printed to stdout.
- With `--output`, the JUnit XML is also written — ideal for CI reporters
  (see [GitHub Actions](../integrations/github-actions.md)).
- **Exit code** is `0` when every test passed, `1` otherwise. A broken
  invocation (bad suite file, unreadable model) raises instead.

!!! tip "Report formats in Python code"

    The CLI always prints the table view. From Python you can render JSON or
    JUnit XML for any `SuiteResult` — see [Reports](core-concepts.md#reports).

## Typical output

```text
Suite: Income CLI Demo

STATUS   TEST                                TIME (ms)    DETAIL
--------------------------------------------------------------------------------
PASS     MinimumAccuracyTest                 8.2
PASS     GroupPerformanceTest                9.1
PASS     RobustnessTest                      21.7
FAIL     DataDriftTest                       3.4          PSI for income = 0.4102 exceeds max_psi 0.1
--------------------------------------------------------------------------------
3 passed, 1 failed
```
