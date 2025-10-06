from __future__ import annotations

# Entrypoint to register Lab 1 DAGs with Airflow
# We import the factory function from the module-level DAG file to keep parity
# with the reference layout, while keeping our custom DAG implementation.

from wine_quality_ml_dag import dag as wine_quality_ml  # noqa: F401

# If you later add more DAGs for Lab 1, import them here similarly so Airflow
# discovers them via this single entrypoint file.


