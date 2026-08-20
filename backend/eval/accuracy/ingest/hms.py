"""NOAA HMS historical smoke-polygon ingest adapter.

NOAA's Hazard Mapping System publishes analyst-drawn daily smoke-plume
polygons. This adapter ingests the historical archive on NESDIS
(``https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/hms_backup``): for each
day it tries the daily smoke KML first (``.../KML/smoke<YYYYMMDD>.kml``) and
falls back to the shapefile zip (``.../GIS/SMOKE/hms_smoke<YYYYMMDD>.zip``).
For 2003-2019 those per-day files are NOT published — the years exist only as
yearly ``<YYYY>.tar`` tarballs at the top of ``hms_backup/`` — so the adapter
ingests those years from the tarball (the daily smoke KMLs live inside as
``KML/smoke<YYYYMMDD>.kml.gz``) and keeps the per-day path for 2020+.
Only the KML path is parsed — with the stdlib ``xml.etree.ElementTree`` plus
``tarfile``/``gzip``, so no new dependency is required; a day that resolves to
no KML is skipped.

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
- 2003-2019 exist only as ``<YYYY>.tar`` tarballs at the top of ``hms_backup/``
  (``<YYYY>.tar.gz`` does NOT resolve). Each tarball mirrors the year's whole
  ``hms_backup/<YYYY>/`` tree; the analyst smoke KMLs are stored gzip-compressed
  as ``KML/smoke<YYYYMMDD>.kml.gz`` alongside ``fire*``/``hysplit*`` KMLs (a
  separate product, not ingested). Some years' tarballs carry no smoke KMLs at
  all (observed for 2016-2018); those days resolve to no KML and are skipped.
- The 2021+ archive is not hosted on NESDIS (satepsanone stops at 2020).
  ``ingest_hms_smoke_range`` tolerates the resulting 404s by skipping days.
"""

import gzip
import json
import re
import shutil
import tarfile
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import httpx

from backend.eval.accuracy.ingest.base import ArchiveSource
from backend.eval.accuracy.records import HmsSmokeRecord
from backend.eval.accuracy.store import AccuracyStore

HMS_BACKUP_BASE_URL = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/hms_backup"

_TIMEOUT_S = 60.0
# Be polite to the archive server between per-day requests.
_DAY_DELAY_S = 0.2

# Years published ONLY as yearly ``<year>.tar`` tarballs. NESDIS extracts the
# per-day ``hms_backup/<year>/`` tree from 2020 on; 2003-2019 exist only as
# tarballs, so the ingest switches to the tarball path for ``year <=`` this.
YEARLY_TARBALL_LAST_YEAR = 2019

# A whole year's tarball is ~1GB (2019 is 1.14GB), so its download gets a much
# more generous timeout than the small per-day files above.
_TARBALL_TIMEOUT_S = 3600.0

# Basenames of the daily analyst smoke KMLs inside a yearly tarball
# (``smoke<YYYYMMDD>.kml``, gzip-compressed as ``smoke<YYYYMMDD>.kml.gz``).
# ``fire*``/``hysplit*`` KMLs share the tarball's KML directory but are a
# separate product and must not be ingested as smoke records.
_DAILY_SMOKE_KML_RE = re.compile(r"^smoke(\d{8})\.kml(?:\.gz)?$", re.IGNORECASE)

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


def _records_from_day(paths: Iterable[Path], date_local: str) -> List[HmsSmokeRecord]:
    """Parse one day's downloaded files into HMS records.

    Shared by the per-day download path and the yearly-tarball path so both
    ingest through the same parse routine. Only ``*.kml`` files are parsed —
    the shapefile fallback is downloaded, not parsed (KML-only phase). The
    tarball path produces ``smoke<YYYYMMDD>.kml`` files under the same flat
    naming the per-day path writes, so downstream handling is identical.
    """
    records: List[HmsSmokeRecord] = []
    for path in paths:
        if path.suffix.lower() != ".kml":
            continue
        records.extend(parse_hms_kml(path.read_text(encoding="utf-8"), date_local))
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


def _download_year_tarball(year: int, tar_path: Path) -> bool:
    """Download ``hms_backup/<year>.tar`` (with a ``.tar.gz`` fallback) into
    ``tar_path``, streaming the ~1GB body to disk.

    Returns True on success; False when the archive does not exist (both
    candidates 404), so the caller can fall back to the per-day URLs. The
    body is written to a ``.part`` sibling first and renamed only once
    complete, so an interrupted download never leaves a corrupt tarball
    behind. Network errors and unexpected statuses propagate — the CLI then
    fails the year chunk and keeps the watermark from advancing past it.
    """
    candidates = [
        f"{HMS_BACKUP_BASE_URL}/{year}.tar",
        f"{HMS_BACKUP_BASE_URL}/{year}.tar.gz",
    ]
    part_path = tar_path.parent / f"{tar_path.name}.part"
    with httpx.Client(timeout=_TARBALL_TIMEOUT_S, follow_redirects=True) as client:
        for url in candidates:
            try:
                with client.stream("GET", url) as resp:
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    with open(part_path, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
                    part_path.rename(tar_path)
                    return True
            except httpx.HTTPError:
                part_path.unlink(missing_ok=True)
                raise
    return False


def _extract_tarball_kmls(tar_path: Path, out_dir: Path) -> List[Path]:
    """Extract the daily smoke KMLs from a yearly HMS tarball into out_dir.

    The yearly tarball mirrors the whole ``hms_backup/<year>/`` directory tree
    (auto-detection products, fire/hysplit KMLs, shapefiles) and stores the
    analyst smoke KMLs gzip-compressed (``KML/smoke<YYYYMMDD>.kml.gz``). Only
    members whose basename matches ``_DAILY_SMOKE_KML_RE`` are written, each
    decompressed (gzip when needed) to ``out_dir/smoke<YYYYMMDD>.kml`` — the
    same flat naming the per-day path produces, so downstream parsing is
    identical. Members are selected by basename so a path-traversal member
    name cannot escape out_dir, and extraction is bounded by the same
    ``MAX_EXTRACT_MEMBERS`` / ``MAX_EXTRACT_BYTES`` guards as ``_extract_zip``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Path] = []
    extracted_bytes = 0
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            match = _DAILY_SMOKE_KML_RE.match(Path(member.name).name)
            if match is None:
                continue
            if len(extracted) >= MAX_EXTRACT_MEMBERS:
                raise RuntimeError(
                    f"{tar_path.name}: archive has more than {MAX_EXTRACT_MEMBERS} "
                    "smoke KML members"
                )
            extracted_bytes += member.size
            if extracted_bytes > MAX_EXTRACT_BYTES:
                raise RuntimeError(
                    f"{tar_path.name}: decompressed size exceeds "
                    f"{MAX_EXTRACT_BYTES} bytes"
                )
            target = out_dir / f"smoke{match.group(1)}.kml"
            src = tf.extractfile(member)
            if Path(member.name).name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=src, mode="rb") as gz, open(target, "wb") as dst:
                    shutil.copyfileobj(gz, dst)
            else:
                with open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def fetch_hms_smoke_year_tarball(year: int, out_dir: Path) -> Optional[List[Path]]:
    """Download the yearly ``<year>.tar`` tarball for 2003-2019 and extract
    its daily smoke KMLs, returning the extracted ``smoke<YYYYMMDD>.kml``
    paths.

    Returns ``None`` when the year has no tarball (both ``<year>.tar`` and
    ``<year>.tar.gz`` 404) so callers can fall back to the per-day URLs, an
    empty list when the tarball exists but carries no smoke KMLs (a year that
    truly published none, observed for 2016-2018), or the extracted paths.
    Resumable like the AQS adapter: an already-downloaded tarball in out_dir
    is reused instead of re-fetching the ~1GB archive.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{year}.tar"

    if tar_path.exists():
        if tar_path.stat().st_size == 0:
            return []
    elif not _download_year_tarball(year, tar_path):
        return None
    return _extract_tarball_kmls(tar_path, out_dir)


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

    Years through ``YEARLY_TARBALL_LAST_YEAR`` (2003-2019) are ingested from
    the yearly ``<year>.tar`` tarball: it is fetched/extracted once per year
    and each day's ``smoke<YYYYMMDD>.kml`` is parsed from it. When the tarball
    is missing for the year the per-day URLs are tried instead; when the
    tarball exists but the day (or year) has no smoke KML, the day is skipped.
    Years after that (2020+) use the per-day URLs directly. Days whose KML
    (or zip fallback) is missing are skipped rather than erroring, and a short
    sleep keeps the archive server polite between days. Returns the number of
    records written. Shapefile-zip fallbacks are downloaded but not parsed in
    this KML-only phase.
    """
    out_dir = Path(out_dir)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    total = 0
    first = True
    day = start
    year_kml_paths: Dict[int, Optional[List[Path]]] = {}
    while day <= end:
        if not first:
            time.sleep(_DAY_DELAY_S)
        first = False

        year = day.year
        day_kml_name = f"smoke{day.strftime('%Y%m%d')}.kml"

        day_paths: List[Path] = []
        if year <= YEARLY_TARBALL_LAST_YEAR:
            if year not in year_kml_paths:
                year_kml_paths[year] = fetch_hms_smoke_year_tarball(year, out_dir)
            tarball_kmls = year_kml_paths[year]
            if tarball_kmls is not None:
                day_paths = [p for p in tarball_kmls if p.name == day_kml_name]
            if not day_paths and tarball_kmls is None:
                # No tarball for the year — fall back to the per-day URLs,
                # which skip the day when both 404.
                day_paths = fetch_hms_smoke_daily(day, out_dir)
        else:
            day_paths = fetch_hms_smoke_daily(day, out_dir)

        total += store.insert_hms_smoke(_records_from_day(day_paths, day.isoformat()))
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
