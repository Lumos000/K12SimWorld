# Contributing

Contributions should preserve K12SimWorld's central rule: a generated visual is
not evidence of a correct simulation unless it is backed by an executable,
auditable state trace.

Before opening a pull request:

1. Add or update deterministic tests for solver, schema, routing, or validation changes.
2. Keep API keys, datasets, model responses, videos, and machine-specific paths out of Git.
3. Run `python -m unittest discover -s tests -p "test_*.py" -v`.
4. Document new `simulation_spec` fields in the JSON Schema and Chinese solver guide.
5. State numerical assumptions and unsupported regimes explicitly.

Generated benchmark results should report failures rather than silently dropping them.
