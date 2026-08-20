import csv
import json

from backend import db as db_module
from backend import human_labels as labels_module


def _seed_cache(db_path):
    db_module.init_db()
    for key, verdict in [
        ("why_1", {"verdict": "pass", "judge_model": "model-a"}),
        ("why_2", {"verdict": "fail", "judge_model": "model-a"}),
        ("why_3", {"verdict": "pass", "judge_model": "model-a"}),
    ]:
        db_module.set_cached_narrative(
            key,
            f"narrative for {key}",
            {"location": {"name": "Test"}},
            verdict,
            # override db path by monkeypatching below
        )


def test_export_then_validate(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    monkeypatch.setattr(labels_module, "DB_PATH", str(db_path))
    _seed_cache(db_path)

    out_csv = tmp_path / "labels.csv"
    assert labels_module.export_rows(str(db_path), str(out_csv), limit=10) == 3

    # Fill in human labels: agree on two, disagree on one.
    rows = list(csv.DictReader(open(out_csv)))
    by_key = {r["cache_key"]: r for r in rows}
    by_key["why_1"]["human_label"] = "pass"
    by_key["why_2"]["human_label"] = "pass"
    by_key["why_3"]["human_label"] = "fail"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = labels_module.validate_labels(str(out_csv), str(db_path))
    assert result["total_cases"] == 3
    assert result["judged_cases"] == 3
    assert result["exact_agreement"] == round(1 / 3, 4)
    assert result["confusion"]["gold_pass_judge_pass"] == 1
    assert result["confusion"]["gold_fail_judge_fail"] == 0


def test_skip_rows_are_excluded(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    monkeypatch.setattr(labels_module, "DB_PATH", str(db_path))
    _seed_cache(db_path)

    out_csv = tmp_path / "labels.csv"
    labels_module.export_rows(str(db_path), str(out_csv), limit=10)
    rows = list(csv.DictReader(open(out_csv)))
    by_key = {r["cache_key"]: r for r in rows}
    by_key["why_1"]["human_label"] = "pass"
    by_key["why_2"]["human_label"] = "skip"
    by_key["why_3"]["human_label"] = "pass"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = labels_module.validate_labels(str(out_csv), str(db_path))
    assert result["total_cases"] == 3
    assert result["judged_cases"] == 2
    assert result["exact_agreement"] == 1.0


def test_cli_export_and_validate(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    monkeypatch.setattr(labels_module, "DB_PATH", str(db_path))
    _seed_cache(db_path)

    out_csv = tmp_path / "labels.csv"
    assert labels_module.main(["--db", str(db_path), "export", "--out", str(out_csv), "--limit", "10"]) == 0
    rows = list(csv.DictReader(open(out_csv)))
    for row in rows:
        row["human_label"] = "pass"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    out_json = tmp_path / "validation.json"
    assert labels_module.main(
        ["--db", str(db_path), "validate", "--labels", str(out_csv), "--out", str(out_json)]
    ) == 0
    data = json.loads(out_json.read_text())
    assert data["judged_cases"] == 3
