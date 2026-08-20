"""EPA AirData PM2.5 speciation daily ingest adapter.

EPA publishes pre-generated, no-key speciation files at
``https://aqs.epa.gov/aqsweb/airdata/daily_SPEC_<year>.zip`` (same bulk format
as the ``daily_<param>_<year>.zip`` AQS daily summaries). Each zip contains one
CSV; this module parses it into canonical ``SpeciationRow`` objects and
persists them through ``AccuracyStore``.

Parsing is header-NAME-based (column index resolved by name, not position),
tolerant to missing columns and quoted commas, and STREAMS the CSV so the
multi-hundred-MB decompressed file is never held in memory whole. Rows are
kept only when they fall inside the Western-US bounding box (to keep the table
small); every parameter/method inside the box is stored (audit-complete), and
the IMPROVE-only filter that labels use is applied later by
``speclabels.derive_components`` via the stored ``method_name``.
"""

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import httpx

from backend.eval.accuracy.ingest.aqs import AQS_BASE_URL, _extract_csvs
from backend.eval.accuracy.records import SpeciationRow
from backend.eval.accuracy.store import AccuracyStore

# Western-US bbox: west, south, east, north. Keeps the speciation table small
# while covering the West Coast + Intermountain West fire/dust regions.
WESTERN_BBOX: Tuple[float, float, float, float] = (-125.0, 31.0, -102.0, 49.0)

_DOWNLOAD_TIMEOUT_S = 120.0

# Speciation rows are inserted (and committed) in batches of this many, so a
# full-year ingest only holds one batch in memory at a time.
INSERT_BATCH_SIZE = 50_000

# Header columns (order as published by EPA) we map onto SpeciationRow.
_REQUIRED_COLUMNS = ("state code", "county code", "site num", "parameter code", "date local")
_OPTIONAL_COLUMNS = {
    "parameter_name": "parameter name",
    "lat": "latitude",
    "lon": "longitude",
    "concentration": "arithmetic mean",
    "units": "units of measure",
    "method_code": "method code",
    "method_name": "method name",
}


def _column_index(header: List[str], name: str) -> Optional[int]:
    """Index of ``name`` in a lowercased header, or None when absent."""
    try:
        return header.index(name)
    except ValueError:
        return None


def _cell(row: List[str], idx: Optional[int]) -> Optional[str]:
    """Stripped cell value or None when the column is absent/empty."""
    if idx is None or idx >= len(row):
        return None
    value = row[idx].strip()
    return value if value else None


def _as_float(cell: Optional[str]) -> Optional[float]:
    if cell is None:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _format_site_id(state_code: str, county_code: str, site_num: str) -> str:
    """Canonical zero-padded site id: ``SS-CCC-NNNN`` (e.g. 06-037-0002) —
    identical to ``backend.eval.accuracy.ingest.aqs``."""
    return f"{state_code.zfill(2)}-{county_code.zfill(3)}-{site_num.zfill(4)}"


def iter_speciation_rows(
    csv_path: Path,
    bbox: Tuple[float, float, float, float] = WESTERN_BBOX,
) -> Iterator[SpeciationRow]:
    """Stream-parse a speciation CSV into ``SpeciationRow`` objects.

    Column lookup is by header name (order-independent) and tolerates missing
    optional columns. Rows lacking natural-key columns, a numeric
    concentration, or coordinates inside ``bbox`` are skipped. The file is
    read incrementally (one csv row at a time), so an archive-sized CSV can be
    consumed without loading it whole.
    """
    west, south, east, north = bbox
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            raw_header = next(reader)
        except StopIteration:
            return
        # Tolerate a UTF-8 BOM on the first header cell.
        header = [cell.strip().strip("\ufeff").lower() for cell in raw_header]

        col = {name: _column_index(header, csv_col) for name, csv_col in _OPTIONAL_COLUMNS.items()}
        required = {name: _column_index(header, csv_col) for name, csv_col in
                    zip(("state_code", "county_code", "site_num", "parameter_code", "date_local"),
                        _REQUIRED_COLUMNS)}

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue

            state_code = _cell(row, required["state_code"])
            county_code = _cell(row, required["county_code"])
            site_num = _cell(row, required["site_num"])
            parameter_code = _cell(row, required["parameter_code"])
            date_local = _cell(row, required["date_local"])
            if None in (state_code, county_code, site_num, parameter_code, date_local):
                continue

            lat = _as_float(_cell(row, col["lat"]))
            lon = _as_float(_cell(row, col["lon"]))
            # Keep the table small: rows outside the Western-US box are dropped
            # outright, before parsing the concentration.
            if lat is None or lon is None:
                continue
            if not (west <= lon <= east and south <= lat <= north):
                continue

            concentration = _as_float(_cell(row, col["concentration"]))
            if concentration is None:
                continue

            yield SpeciationRow(
                site_id=_format_site_id(state_code, county_code, site_num),
                date_local=date_local,
                parameter_code=parameter_code,
                parameter_name=_cell(row, col["parameter_name"]),
                method_code=_cell(row, col["method_code"]),
                method_name=_cell(row, col["method_name"]),
                concentration=concentration,
                units=_cell(row, col["units"]),
                lat=lat,
                lon=lon,
            )


def parse_speciation_csv(
    csv_path: Path,
    bbox: Tuple[float, float, float, float] = WESTERN_BBOX,
) -> List[SpeciationRow]:
    """Materialize ``iter_speciation_rows`` into a list (tests / small files)."""
    return list(iter_speciation_rows(csv_path, bbox))


def download_speciation_year(year: int, out_dir: Path) -> List[Path]:
    """Download ``daily_SPEC_<year>.zip`` into ``out_dir`` and extract it,
    returning the extracted CSV paths.

    Resumable: an already-downloaded zip is skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"daily_SPEC_{year}.zip"
    zip_path = out_dir / zip_name
    if not zip_path.exists():
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
            resp = client.get(f"{AQS_BASE_URL}/{zip_name}")
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
    return _extract_csvs(zip_path, out_dir)


def ingest_speciation_year(
    year: int,
    store: AccuracyStore,
    out_dir: Path,
    bbox: Tuple[float, float, float, float] = WESTERN_BBOX,
) -> int:
    """Download, stream-parse, and persist the speciation year.

    Returns the number of records written to the store. Parsing streams the
    CSV (see ``iter_speciation_rows``) and inserts in ``INSERT_BATCH_SIZE``
    batches, so memory stays flat regardless of file size. Distinct
    ``(site_id, lat, lon)`` sites are also persisted into the
    ``speciation_sites`` table (the SPEC CSV's Latitude/Longitude columns) so
    ``ingest weather --speciation`` can fetch weather for them; only sites
    inside ``bbox`` contribute, matching the speciation rows that were kept.
    """
    csv_paths = download_speciation_year(year, out_dir)
    total = 0
    batch: List[SpeciationRow] = []
    sites: Dict[str, Tuple[float, float]] = {}
    for csv_path in csv_paths:
        for record in iter_speciation_rows(csv_path, bbox=bbox):
            if record.lat is not None and record.lon is not None:
                sites.setdefault(record.site_id, (record.lat, record.lon))
            batch.append(record)
            if len(batch) >= INSERT_BATCH_SIZE:
                total += store.insert_speciation(batch)
                batch = []
    if batch:
        total += store.insert_speciation(batch)
    store.insert_speciation_sites(
        (site_id, lat, lon) for site_id, (lat, lon) in sorted(sites.items())
    )
    return total
