"""NOAA HMS historical smoke-polygon ingest adapter.

NOAA's Hazard Mapping System publishes analyst-drawn daily smoke-plume
polygons. This adapter ingests the historical archive on NESDIS
(``https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/hms_backup``): for each
day it tries the daily smoke KML first (``.../KML/smoke<YYYYMMDD>.kml``) and
falls back to the shapefile zip (``.../GIS/SMOKE/hms_smoke<YYYYMMDD>.zip``).
Only the KML path is parsed — with the stdlib ``xml.etree.ElementTree``, so no
new dependency is required; a day that resolves to neither URL is skipped.

KML parsing is namespace-tolerant: NOAA's files carry the KML 2.2 default
namespace (``http://www.opengis.net/kml/2.2``) and bare/unnamespaced XML is
accepted too. Density is read from the placemark ``<name>``/``<description>``
by whole-word match ("light"/"medium"/"heavy"), or from the numeric code the
description actually carries (``Density: 5|16|27`` -> light|medium|heavy),
defaulting to "light" when nothing is found.

Archive layout (verified 2026-08):
- ``hms_backup/2020/`` is extracted; ``KML/smoke<YYYYMMDD>.kml`` and
  ``GIS/SMOKE/hms_smoke<YYYYMMDD>.zip`` are both live per-day URLs (the
  ``hms_smoke<YYYYMMDD>.kml`` name does NOT resolve; NOAA names the smoke KMLs
  ``smoke<YYYYMMDD>.kml``).
- 2003-2019 exist only as ``<YYYY>.tar`` tarballs at the top of ``hms_backup/``,
  so per-day KML/zip URLs 404 for those years (known gap for this phase).
- The 2021+ archive is not hosted on NESDIS (satepsanone stops at 2020).
  ``ingest_hms_smoke_range`` tolerates the resulting 404s by skipping days.
"""

import json
import re
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import httpx

from backend.eval.accuracy.ingest.base import ArchiveSource
from backend.eval.accuracy.records import HmsSmokeRecord
from backend.eval.accuracy.store import AccuracyStore

HMS_BACKUP_BASE_URL = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/hms_backup"

_TIMEOUT_S = 60.0
# Be polite to the archive server between per-day requests.
_DAY_DELAY_S = 0.2

# Zip-bomb guards: total decompressed bytes and member count are capped when
# unzipping archive downloads (see ``_extract_zip``).
MAX_EXTRACT_BYTES = 2_000_000_000
MAX_EXTRACT_MEMBERS = 10_000

# Numeric density codes as published in the KML descriptions (mirrors the live
# HMS feed codes in backend.services.hms) plus the word forms used in labels.
_DENSITY_BY_CODE = {"5": "light", "16": "medium", "27": "heavy"}
_DENSITY_WORDS = ("light", "medium", "heavy")


def _local_name(tag: str) -> str:
    """Local XML tag name, ignoring any ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def _descendant_text(elem: ET.Element, name: str) -> Optional[str]:
    """Text of the first descendant element whose local name is ``name``."""
    child = next((e for e in elem.iter() if _local_name(e.tag) == name), None)
    if child is None or child.text is None:
        return None
    return child.text


def _parse_coordinates(text: str) -> List[List[float]]:
    """Parse a KML ``<coordinates>`` body into ``[[lon, lat], ...]``.

    Each whitespace-separated tuple is ``lon,lat[,alt]``; the altitude is
    dropped. Unparseable tuples are skipped.
    """
    points: List[List[float]] = []
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        try:
            points.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    return points


def _extract_density(name_text: Optional[str], description_text: Optional[str]) -> str:
    """Density label from a placemark's name/description.

    Prefers the numeric code the descriptions actually carry (``Density: 5``
    etc.), then falls back to whole-word "light"/"medium"/"heavy", then to
    "light". Matching is case/whitespace tolerant.
    """
    haystack = f"{name_text or ''}\n{description_text or ''}".lower()
    code_match = re.search(r"density\s*:?\s*(\d+)", haystack)
    if code_match:
        label = _DENSITY_BY_CODE.get(code_match.group(1))
        if label is not None:
            return label
    for word in _DENSITY_WORDS:
        if re.search(rf"\b{word}\b", haystack):
            return word
    return "light"


def parse_hms_kml(kml_text: str, date_local: str) -> List[HmsSmokeRecord]:
    """Parse daily HMS smoke KML into one record per smoke polygon.

    Every ``<Placemark>`` carrying a ``<Polygon>`` yields a record whose
    ``geometry_json`` is a GeoJSON Polygon geometry (exterior ring first,
    ``<innerBoundaryIs>`` rings appended as holes). Placemarks without a
    polygon (points, markers) and polygons without coordinates are skipped.
    Handles both the KML 2.2 default namespace and bare XML.
    """
    try:
        root = ET.fromstring(kml_text)
    except ET.ParseError:
        return []

    records: List[HmsSmokeRecord] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "Placemark":
            continue

        polygon = next((e for e in elem.iter() if _local_name(e.tag) == "Polygon"), None)
        if polygon is None:
            continue  # e.g. a point/marker placemark

        outer = next(
            (e for e in polygon.iter() if _local_name(e.tag) == "outerBoundaryIs"),
            None,
        )
        if outer is None:
            continue
        outer_ring = _parse_coordinates(_descendant_text(outer, "coordinates") or "")
        if not outer_ring:
            continue

        rings = [outer_ring]
        for inner in (e for e in polygon.iter() if _local_name(e.tag) == "innerBoundaryIs"):
            hole = _parse_coordinates(_descendant_text(inner, "coordinates") or "")
            if hole:
                rings.append(hole)

        density = _extract_density(
            _descendant_text(elem, "name"),
            _descendant_text(elem, "description"),
        )
        records.append(HmsSmokeRecord(
            date_local=date_local,
            density=density,
            geometry_json=json.dumps(
                {"type": "Polygon", "coordinates": rings}, separators=(",", ":")
            ),
        ))
    return records


def _extract_zip(zip_path: Path, out_dir: Path) -> List[Path]:
    """Extract every member of a downloaded HMS zip into out_dir (flattened to
    the member basename, matching the AQS adapter) and return the paths.

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
            if not name:
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


def fetch_hms_smoke_daily(date: datetime, out_dir: Path) -> List[Path]:
    """Download one day of HMS smoke polygons into ``out_dir``.

    Tries the daily smoke KML first (``.../KML/smoke<YYYYMMDD>.kml``) and falls
    back to the shapefile zip (``.../GIS/SMOKE/hms_smoke<YYYYMMDD>.zip``) when
    the KML is missing. KML files are written as-is; zips are extracted and the
    member paths returned. Returns ``[]`` when neither URL resolves. Only the
    KML path is parsed downstream (see :func:`ingest_hms_smoke_range`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ymd = date.strftime("%Y%m%d")
    year = date.strftime("%Y")
    kml_url = f"{HMS_BACKUP_BASE_URL}/{year}/KML/smoke{ymd}.kml"
    zip_url = f"{HMS_BACKUP_BASE_URL}/{year}/GIS/SMOKE/hms_smoke{ymd}.zip"

    with httpx.Client(timeout=_TIMEOUT_S, follow_redirects=True) as client:
        try:
            resp = client.get(kml_url)
        except httpx.HTTPError:
            resp = None
        if resp is not None and resp.status_code == 200:
            kml_path = out_dir / f"smoke{ymd}.kml"
            kml_path.write_bytes(resp.content)
            return [kml_path]

        try:
            resp = client.get(zip_url)
        except httpx.HTTPError:
            resp = None
        if resp is not None and resp.status_code == 200:
            zip_path = out_dir / f"hms_smoke{ymd}.zip"
            zip_path.write_bytes(resp.content)
            return _extract_zip(zip_path, out_dir)
    return []


def ingest_hms_smoke_range(
    start_date: str,
    end_date: str,
    store: AccuracyStore,
    out_dir: Path,
) -> int:
    """Download, parse, and persist HMS smoke polygons for every day in
    ``[start_date, end_date]`` (both ``YYYY-MM-DD``).

    Days whose KML (or zip fallback) is missing are skipped rather than
    erroring, and a short sleep keeps the archive server polite between days.
    Returns the number of records written. Shapefile-zip fallbacks are
    downloaded but not parsed in this KML-only phase.
    """
    out_dir = Path(out_dir)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    total = 0
    first = True
    day = start
    while day <= end:
        if not first:
            time.sleep(_DAY_DELAY_S)
        first = False

        for path in fetch_hms_smoke_daily(day, out_dir):
            if path.suffix.lower() != ".kml":
                continue  # shapefile fallback is downloaded, not parsed (KML phase)
            records = parse_hms_kml(path.read_text(encoding="utf-8"), day.isoformat())
            total += store.insert_hms_smoke(records)
        day += timedelta(days=1)
    return total


class HmsSmokeSource(ArchiveSource[HmsSmokeRecord]):
    """ArchiveSource wrapper around :func:`ingest_hms_smoke_range` for uniform
    pipeline use."""

    source_name = "hms_smoke"

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)

    def fetch(self, year: int, store: AccuracyStore, **kwargs) -> List[HmsSmokeRecord]:
        ingest_hms_smoke_range(f"{year}-01-01", f"{year}-12-31", store, self.out_dir)
        return [r for r in store.fetch_hms_smoke() if r.date_local.startswith(f"{year}-")]
