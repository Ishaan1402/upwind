"""Offline accuracy-evaluation pipeline for the Upwind attribution scorer.

Phase 1 backtests the attribution scorer against historical air-quality and
fire data. This package ingests canonical historical records (AQS daily
summaries first), persists them to SQLite, and will later derive labels and
score against them (fire/smoke/weather adapters, label derivation, and
metrics land in later chunks).
"""

from pathlib import Path

from backend.eval.accuracy.records import AqsDailyRecord
from backend.eval.accuracy.store import AccuracyStore

# Standardized raw-download root: ``data/raw/<source>/<year>/<filename>``
# (e.g. ``data/raw/aqs/2020/daily_88101_2020.zip``). The SQLite store lives one
# level up (``data/accuracy.db``) so a full wipe of downloaded raw data never
# touches ingested state.
RAW_DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

__all__ = ["AqsDailyRecord", "AccuracyStore", "RAW_DATA_DIR"]
