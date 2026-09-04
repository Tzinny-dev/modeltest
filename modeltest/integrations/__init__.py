"""Integration adapters between modeltest and external tooling.

These modules are optional: importing them does not pull in heavy
dependencies. Each adapter pulls in its framework lazily inside the functions
that need it, so ``pip install modeltest[mlflow]`` is only required when you
actually use the integration.
"""
