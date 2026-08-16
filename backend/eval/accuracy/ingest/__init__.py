"""Data ingest adapters for the accuracy-evaluation pipeline.

Each adapter turns a historical data source (AQS daily summaries, and later
FIRMS-historical, HMS, NIFC, weather) into canonical records persisted through
:class:`backend.eval.accuracy.store.AccuracyStore`.
"""

from backend.eval.accuracy import RAW_DATA_DIR

__all__ = ["RAW_DATA_DIR"]
