from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from backend.routers import aqi, why, events
from backend.config import PORT, CORS_ORIGINS
from backend.db import init_db
from backend.middleware.rate_limit import RateLimitAndLoggingMiddleware

# Initialize SQLite database for narrative caching & geocoding cache
init_db()

app = FastAPI(
    title="AQI 'Show Why' API",
    description="Backend API for US Air Quality Index observation and evidence-first attribution.",
    version="1.0.0"
)

# Add Rate Limiting & Light Request Logging Middleware
app.add_middleware(RateLimitAndLoggingMiddleware)

# Enable CORS driven by environment config
app.add_middleware(CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(aqi.router)
app.include_router(why.router)
app.include_router(events.router)

@app.get("/health")
async def health():
    return {"status": "ok", "app": "AQI Show Why MVP"}

# Production single-process SPA static serving
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if DIST_DIR.exists() and DIST_DIR.is_dir():
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        clean_path = full_path.lower().strip()
        
        # Block API routes, sensitive files, dotfiles, and backup configurations with strict 404
        if (
            clean_path.startswith("api/") or clean_path == "api" or clean_path == "health" or
            ".env" in clean_path or ".bak" in clean_path or ".git" in clean_path or
            "config" in clean_path or clean_path.startswith(".") or
            any(clean_path.endswith(ext) for ext in [".env", ".bak", ".config", ".key", ".pem", ".db", ".json", ".ini"])
        ):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        
        # Resolve against the dist root and require containment so traversal
        # segments (e.g. /assets/../../backend/metrics.py) can never escape.
        dist_root = DIST_DIR.resolve()
        target_file = (dist_root / full_path).resolve()
        if target_file.is_file() and target_file.is_relative_to(dist_root):
            return FileResponse(target_file)
        
        index_file = DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(status_code=404, content={"detail": "Index file missing"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=PORT, reload=True)
