# Upwind

[![Deploy Upwind](https://github.com/Ishaan1402/upwind/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ishaan1402/upwind/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-getupwind.me-blue?style=flat-square)](https://getupwind.me)

An AQI map for the US that explains why the air is bad instead of just displaying a number. Upwind breaks down local air quality spikes, pulling satellite feeds, sensor data, and current events to rank the most likely causes.

| Class | Likely Triggers |
|---|---|
| **Wildfire Smoke** | PM2.5 + satellite fire hotspots |
| **Ozone Episode** | High ground O3 + high heat (≥75°F) |
| **Windblown Dust** | High coarse PM10 + high winds |
| **Thermal Inversion** | Fine PM2.5 + cold, calm weather + shallow mixing layer |
| **Urban Pollution** | Local PM2.5 + clean overhead AOD + no fire hotspots |

> **Note**: Upwind provides evidence and heuristics based on public satellite and monitor feeds. It is **not** physical air sampling or lab testing. The application's conclusions are ranked observational hypotheses, not verdicts.

---

## Stack
- **Backend:** Python 3.11, FastAPI, SQLite
- **Frontend:** React 18, Vite, MapLibre GL / Leaflet
- **Data Sources:** AirNow API, Open-Meteo, NASA FIRMS Thermal Hotspots

---

## Environment Variables

Copy `.env.example` to `.env` before running:

| Variable | Type | Description |
|---|---|---|
| `AIRNOW_KEY` | Optional | Primary US monitor data feed. |
| `FIRMS_MAP_KEY` | Optional | NASA satellite thermal hotspot detection. |
| `DEEPSEEK_API_KEY` | Optional | Narrative briefing synthesis. |
| `GROQ_API_KEY` | Optional | LLM Judge verification. |
| `RATE_LIMIT_AQI_PER_MIN` | Optional | Max `/api/aqi` requests/min per IP (default: `60`). |

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