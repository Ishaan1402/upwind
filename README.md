# Upwind

[![Deploy Upwind](https://github.com/Ishaan1402/upwind/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ishaan1402/upwind/actions/workflows/deploy.yml)
[![Narrative Eval](https://github.com/Ishaan1402/upwind/actions/workflows/eval.yml/badge.svg)](https://github.com/Ishaan1402/upwind/actions/workflows/eval.yml)
[![Security Checks](https://github.com/Ishaan1402/upwind/actions/workflows/security.yml/badge.svg)](https://github.com/Ishaan1402/upwind/actions/workflows/security.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-getupwind.me-blue?style=flat-square)](https://getupwind.me)
[![Evidence](https://img.shields.io/badge/Evidence-getupwind.me%2Fevidence-blue?style=flat-square)](https://getupwind.me/evidence/)

An AQI map for the US that explains why the air is bad instead of just displaying a number. Upwind breaks down local air quality spikes, pulling satellite feeds, sensor data, and current events to rank the most likely causes.

## How it's verified

- A nightly GitHub Actions run evaluates 7 fixed scenarios with the LLM judge plus deterministic jargon/header checks, and fails CI if fewer than 4 pass.
- Every judge verdict from live traffic is stored in SQLite. `python -m backend.metrics report` turns request and explanation events into numbers: request counts, p95 latency, cache hit rate, LLM cost per explanation, and judge pass rate.
- `python -m backend.eval_validation` scores the judge against a deterministic grounded/hallucinated set.
- `python -m backend.human_labels export` creates a CSV of recent explanations to label by hand; `python -m backend.human_labels validate` reports agreement and Cohen's kappa in the same JSON the eval dashboard renders.
- The eval dashboard and a public evidence page are published after each nightly run: [https://getupwind.me/evidence/](https://getupwind.me/evidence/).


| Class                 | Likely Triggers                                        |
| --------------------- | ------------------------------------------------------ |
| **Wildfire Smoke**    | PM2.5 + satellite fire hotspots + AOD plumes           |
| **Ozone Episode**     | High ground O3 + high heat (≥75°F)                     |
| **Windblown Dust**    | High coarse PM10 + high winds                          |
| **Thermal Inversion** | Fine PM2.5 + cold, calm weather + shallow mixing layer |
| **Urban Pollution**   | Local PM2.5 + clean overhead AOD + no fire hotspots    |


> **Note**: Upwind provides evidence and heuristics based on public satellite and monitor feeds. It is **not** physical air sampling or lab testing. The application's conclusions are ranked observational hypotheses, not verdicts.

---



## Stack

- **Backend:** Python 3.12, FastAPI, SQLite
- **Frontend:** React 18, Vite, MapLibre GL / Leaflet

## Global coverage

The app accepts any location, but data quality differs by country. The API
returns a `coverage` object so clients can be explicit about the mode:

| Mode | When | Data sources |
| ---- | ---- | ------------ |
| `us` | US ZIP/location or AirNow-confirmed | AirNow, Open-Meteo, OpenAQ (US), FIRMS |
| `international` | Non-US country code known | Open-Meteo, OpenAQ (country), FIRMS; no AirNow |
| `unknown` | Country could not be confirmed | Best-effort global feeds with a disclaimer |

Outside the US, the AQI is the US-style index from Open-Meteo, not a local
regulatory index. Attribution remains an evidence-based hypothesis.

## Metrics & evidence

Every HTTP request is recorded in SQLite (`backend/cache.db`), and the nightly
eval workflow publishes a public scorecard at
[https://getupwind.me/evidence/](https://getupwind.me/evidence/) with request
counts, p95 latency, cache hit rate, LLM cost per explanation, judge pass rate,
and judge validation agreement.

Reproduce locally:

```bash
python -m backend.metrics report --days 30 --out metrics.json
python -m backend.eval_validation --out validation.json
python -m backend.eval render-dashboard --public --metrics metrics.json --validation validation.json --out evidence.html
```

Observation tokens from `/api/aqi` prevent clients from fabricating AQI
observations when `ENFORCE_OBSERVATION_TOKENS=1` is set in production.

---

## Acknowledgements

- Air quality index: U.S. EPA [AirNow](https://www.airnow.gov/)
- Monitor concentrations: [OpenAQ](https://openaq.org/)
- Weather & aerosol data: [Open-Meteo.com](https://open-meteo.com) 
- Fire hotspots: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (LANCE/ESDIS)

---



## Environment Variables

Copy `.env.example` to `.env` before running:


| Variable                 | Type     | Description                              |
| ------------------------ | -------- | ---------------------------------------- |
| `AIRNOW_KEY`             | Optional | Primary US monitor data feed             |
| `OPENAQ_API_KEY`         | Optional | US reference-monitor concentrations (OpenAQ v3) |
| `FIRMS_MAP_KEY`          | Optional | NASA satellite thermal hotspot detection |
| `DEEPSEEK_API_KEY`       | Optional | My briefing LLM of choice                |
| `GROQ_API_KEY`           | Optional | LLM Judge                                |
| `RATE_LIMIT_AQI_PER_MIN` | Optional | Max `/api/aqi` requests/min per IP       |
| `ENFORCE_OBSERVATION_TOKENS` | Optional | Require signed observation tokens on Why APIs (`1` in production) |
| `OBSERVATION_TOKEN_SECRET` | Optional | HMAC secret for observation tokens       |
| `LLM_INPUT_PRICE_PER_1M` | Optional | Cost estimate input price (USD / 1M tokens) |
| `LLM_OUTPUT_PRICE_PER_1M` | Optional | Cost estimate output price (USD / 1M tokens) |


---



## Local Dev



### 1. Backend API

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```



### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`
