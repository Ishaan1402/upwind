"""EPA AQS (Air Quality System) daily summary ingest adapter.

EPA publishes pre-generated, no-key daily summary files at
``https://aqs.epa.gov/aqsweb/airdata/daily_<param>_<year>.zip`` plus a site
coordinate snapshot (``aqs_sites.zip``). Each zip contains one CSV; this
module parses those CSVs into canonical ``AqsDailyRecord`` objects and
persists them through ``AccuracyStore``.

Parsing is header-NAME-based (column index resolved by name, not position) and
tolerant to missing columns and quoted commas. Raw values are stored faithfully
— no unit or concentration conversion happens here; label derivation
(reconstruct) interprets them later.
"""

import csv
import io
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import httpx

from backend.eval.accuracy.ingest.base import ArchiveSource
from backend.eval.accuracy.records import AqsDailyRecord
from backend.eval.accuracy.store import AccuracyStore

AQS_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"

# Daily summary parameters ingested: PM2.5 FRM/FEM (88101), PM2.5 non-FRM mass
# (88502 — the code IMPROVE speciation sites report PM2.5 mass under), PM10
# (81102), O3 (44201), NO2 (42602), CO (42101), SO2 (42401).
PARAMETER_CODES: List[str] = ["88101", "88502", "81102", "44201", "42602", "42101", "42401"]

_DOWNLOAD_TIMEOUT_S = 60.0

# Zip-bomb guards: total decompressed bytes and member count are capped when
# unzipping archive downloads (see ``_extract_csvs``).
MAX_EXTRACT_BYTES = 2_000_000_000
MAX_EXTRACT_MEMBERS = 10_000

# Header columns (order as published by EPA) we map onto AqsDailyRecord.
_REQUIRED_COLUMNS = ("state code", "county code", "site num", "parameter code", "date local")
_OPTIONAL_COLUMNS = {
    "parameter_name": "parameter name",
    "poc": "poc",
    "lat": "latitude",
    "lon": "longitude",
    "concentration": "arithmetic mean",
    "units": "units of measure",
    "aqi": "aqi",
    "method_code": "method code",
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


def _as_int(cell: Optional[str]) -> Optional[int]:
    if cell is None:
        return None
    try:
        return int(float(cell))
    except ValueError:
        return None


def _as_aqi(cell: Optional[str]) -> Optional[int]:
    """AQI, with empty and negative values normalized to None."""
    value = _as_int(cell)
    if value is None or value < 0:
        return None
    return value


def _format_site_id(state_code: str, county_code: str, site_num: str) -> str:
    """Canonical zero-padded site id: ``SS-CCC-NNNN`` (e.g. 06-037-0002)."""
    return f"{state_code.zfill(2)}-{county_code.zfill(3)}-{site_num.zfill(4)}"


def parse_aqs_daily_csv(csv_text: str) -> List[AqsDailyRecord]:
    """Parse an AQS daily summary CSV into canonical records.

    Column lookup is by header name (order-independent) and tolerates missing
    optional columns. Rows lacking site coordinates or a concentration are
    skipped, as are rows missing natural-key columns.
    """
    reader = csv.reader(io.StringIO(csv_text))
    try:
        raw_header = next(reader)
    except StopIteration:
        return []
    # Tolerate a UTF-8 BOM on the first header cell.
    header = [cell.strip().strip("\ufeff").lower() for cell in raw_header]

    col = {name: _column_index(header, csv_col) for name, csv_col in _OPTIONAL_COLUMNS.items()}
    required = {name: _column_index(header, csv_col) for name, csv_col in
                zip(("state_code", "county_code", "site_num", "parameter_code", "date_local"),
                    _REQUIRED_COLUMNS)}

    records: List[AqsDailyRecord] = []
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
        concentration = _as_float(_cell(row, col["concentration"]))
        # Skip rows without usable coordinates or a measured concentration.
        if lat is None or lon is None or concentration is None:
            continue

        records.append(AqsDailyRecord(
            site_id=_format_site_id(state_code, county_code, site_num),
            state_code=state_code,
            county_code=county_code,
            site_num=site_num,
            parameter_code=parameter_code,
            parameter_name=_cell(row, col["parameter_name"]),
            poc=_as_int(_cell(row, col["poc"])),
            lat=lat,
            lon=lon,
            date_local=date_local,
            concentration=concentration,
            units=_cell(row, col["units"]),
            aqi=_as_aqi(_cell(row, col["aqi"])),
            method_code=_cell(row, col["method_code"]),
        ))
    return records


def _extract_csvs(zip_path: Path, out_dir: Path) -> List[Path]:
    """Extract the CSV members of a downloaded AQS zip into out_dir.

    Members are selected by basename (``Path(member.filename).name``) so a
    path-traversal filename cannot escape out_dir. Extraction is bounded by
    ``MAX_EXTRACT_MEMBERS`` and ``MAX_EXTRACT_BYTES`` (total decompressed size,
    checked against the declared member sizes before any write); exceeding
    either raises a clear error.
    """
    extracted: List[Path] = []
    extracted_bytes = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if not name.lower().endswith(".csv"):
                continue
            if len(extracted) >= MAX_EXTRACT_MEMBERS:
                raise RuntimeError(
                    f"{zip_path.name}: archive has more than {MAX_EXTRACT_MEMBERS} "
                    "extractable members"
                )
            extracted_bytes += member.file_size
            if extracted_bytes > MAX_EXTRACT_BYTES:
                raise RuntimeError(
                    f"{zip_path.name}: decompressed size exceeds "
                    f"{MAX_EXTRACT_BYTES} bytes"
                )
            target = out_dir / name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def download_aqs_year(year: int, out_dir: Path) -> List[Path]:
    """Download the ``daily_<param>_<year>.zip`` files (one per
    ``PARAMETER_CODES``) plus ``aqs_sites.zip`` for ``year`` into ``out_dir``
    and extract them, returning the extracted CSV paths.

    Resumable: an already-downloaded zip is skipped. The sites snapshot is
    fetched from the year-independent ``aqs_sites.zip`` URL but saved as
    ``aqs_sites_<year>.zip`` so per-year out_dirs stay self-contained.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    with httpx.Client(timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
        # (local filename, remote URL)
        archives = [(f"daily_{param}_{year}.zip", f"daily_{param}_{year}.zip") for param in PARAMETER_CODES]
        archives.append((f"aqs_sites_{year}.zip", "aqs_sites.zip"))

        for zip_name, remote_name in archives:
            zip_path = out_dir / zip_name
            if not zip_path.exists():
                resp = client.get(f"{AQS_BASE_URL}/{remote_name}")
                resp.raise_for_status()
                zip_path.write_bytes(resp.content)
            csv_paths.extend(_extract_csvs(zip_path, out_dir))
    return csv_paths


def _parse_year_csvs(csv_paths: Iterable[Path]) -> List[AqsDailyRecord]:
    """Parse all daily-summary CSVs, skipping the sites coordinate snapshot."""
    records: List[AqsDailyRecord] = []
    for csv_path in csv_paths:
        # The sites snapshot has a different schema (no daily readings).
        if csv_path.name.startswith("aqs_sites"):
            continue
        records.extend(parse_aqs_daily_csv(csv_path.read_text(encoding="utf-8", errors="replace")))
    return records


def ingest_aqs_year(year: int, store: AccuracyStore, out_dir: Path) -> int:
    """Download, parse, and persist all daily parameters for ``year``.

    Returns the number of records written to the store.
    """
    csv_paths = download_aqs_year(year, out_dir)
    return store.insert_aqs_daily(_parse_year_csvs(csv_paths))


class AqsDailySource(ArchiveSource[AqsDailyRecord]):
    """ArchiveSource wrapper around :func:`ingest_aqs_year` for uniform pipeline use."""

    source_name = "aqs_daily"

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)

    def fetch(self, year: int, store: AccuracyStore, **kwargs) -> List[AqsDailyRecord]:
        csv_paths = download_aqs_year(year, self.out_dir)
        records = _parse_year_csvs(csv_paths)
        store.insert_aqs_daily(records)
        return records
