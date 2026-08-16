# Wildfire smoke corroboration, corrected wind geometry, and a failure-soft rural prior

- Status: Proposed (implemented; not yet committed)
- Date: 2026-08-12
- Scope: `backend/engine/score.py`, `backend/engine/signals.py`, `backend/services/{firms,hms,wfigs,place_context}.py`, `backend/routers/why.py`, `backend/eval_corpus.py`, and their tests.

## Problem

Attribution had a single source of truth for wildfire smoke — NASA FIRMS thermal
hotspots — and two concrete failure modes:

1. **Wind geometry was inverted.** Open-Meteo reports `wind_direction_10m` as the
   direction the wind comes *from* (meteorological convention), but the code
   treated upwind as `wind_dir + 180`, i.e. the *downwind* direction. This made
   an eastern fire look downwind under an east wind and vice versa.
2. **A single feed outage or mirror retirement silently erased smoke.** If FIRMS
   was unreachable (no key, timeout) or the older WFIGS mirror returned 400,
   real smoke could be scored as "no nearby hotspots" — a false absence claim.
   Separately, a small rural community with elevated PM and no verified fire
   could be crowned "high" generic urban/industrial PM, which is implausible for
   a town with few local sources.

## Decision

This is a bounded attribution calibration, not a rewrite of the scoring engine.
It adds corroborating evidence and corrects geometry while leaving the five
hypothesis model, the router, and the frontend untouched.

### In-scope evidence sources

- NOAA Hazard Mapping System (HMS) smoke-plume analysis — analyst-drawn overhead
  plume polygons (light/medium/heavy density). The raw GeoJSON is never returned;
  only the density verdict is, plus an optional capped polygon list for map layers.
- NIFC WFIGS current federal wildfire incident registry — nearest active named
  wildfire within 300 mi that is not >90% contained. A listing only corroborates
  smoke *transport* when it is upwind-aligned or within 150 mi (parity with the
  FIRMS search cap); a fire farther away and not wind-aligned is disclosed but is
  not transport evidence.
- Census 2020 decennial ZCTA population — a rural/urban prior (population < 5000).
- Corrected Open-Meteo wind geometry — upwind bearing is `wind_dir_deg`
  (the direction the wind comes from), applied consistently in FIRMS, WFIGS, and
  the wind signal.
- FIRMS expanded from VIIRS-SNPP only to all three active VIIRS NRT satellites
  (SNPP/NOAA-20/NOAA-21) over a 48h window, so satellite-pass gaps no longer
  silently drop fires.

### Anti-overclaim rules (the scoring guardrails)

1. **News is never a fire vote on its own.** A news incident name only counts as
   fire evidence when a verified feed (FIRMS, HMS, or WFIGS) corroborates it;
   otherwise it becomes an open question.
2. **Unavailable ≠ absent.** A feed that errored or timed out (`status:
   "unavailable"`) is never phrased as "no hotspots/fires"; it is disclosed as an
   open question instead. Only a genuine `absent` response supports absence claims.
3. **Aloft smoke is never crowned.** HMS/WFIGS present with clean ground PM stays
   `low` (score 30) — smoke overhead that has not settled to breathing level is
   not a high-confidence ground-level event.
4. **Rural prior is failure-soft.** When Census reports a rural ZCTA with elevated
   PM and no verified fire/dust/stagnation, urban/industrial PM is demoted from
   high (75) to a medium/40 baseline. A measured urban tracer (NO2/SO2/CO) or a
   fine-dominated monitor signature can still lift it to medium/55 — specific
   local-combustion evidence — but it never returns to high. A missing Census key
   or network error changes nothing.

### Failure-soft behavior

Every new feed (`fetch_hms_smoke`, `fetch_wfigs_incident`,
`fetch_place_context`) never raises: errors, missing keys, and timeouts return
`status: "unavailable"`, and scoring treats an unavailable signal identically to
a feed that predates the feature. `build_evidence_signals` defaults the three new
signal arguments to `None` (unavailable) so existing callers keep working.

## Non-goals

- No rewrite of `score_hypotheses` ranking, the router architecture, or the
  frontend for style alone.
- No additional feeds, environment variables, tables, or scoring hypotheses
  beyond the five listed above.
- No live-provider or narrative-judge accuracy guarantees: the fixed deterministic
  scenarios are the acceptance gate, not an unrun LLM judge.

## Regression scenarios that pin the behavior

Deterministic scoring tests (`tests/test_engine.py`) plus the eval corpus
(`backend/eval_corpus.py`) cover:

- Government Camp (HMS medium + WFIGS upwind, FIRMS down, rural) → wildfire top/high
  with the incident named, and no "no hotspots" absence claim.
- HMS present + clean ground PM → wildfire stays low (aloft).
- News corroborated by HMS only → a fire vote, no unverified-news open question.
- All three verified feeds unavailable → no absence claims; each outage disclosed.
- Rural unexplained PM → urban capped at medium/40.
- Rural + measured NO2 tracer → urban medium/55, never high.
- FIRMS unavailable → never reads as "no nearby hotspots".
- Wind geometry: a source east of the target is upwind under an east wind and
  downwind under a west wind (FIRMS and WFIGS).

## Limits acknowledged

HMS is coarse, analyst-drawn regional polygons: an HMS hit is treated as
corroborating smoke regardless of exact plume distance. WFIGS corroboration is
gated — a listing only counts as transport evidence when upwind-aligned or within
150 mi, matching the FIRMS search cap. Within a corroborating feed, the exact
distance or density is not further differentiated, which is intentionally coarse;
tightening it further without targeted regression coverage would overfit or
discard valuable regional evidence.
