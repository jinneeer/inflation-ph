"""FastAPI app factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.db import connect
from app.routers import forecast, meta, news, panels, story

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Philippine CPI Analytics", version="1.0.0",
              description="PSA OpenSTAT inflation analytics and forecasting API")

connect()

app.include_router(meta.router, prefix="/api", tags=["meta"])
app.include_router(panels.router, prefix="/api", tags=["panels"])
app.include_router(forecast.router, prefix="/api", tags=["models"])
app.include_router(news.router, prefix="/api", tags=["news"])
app.include_router(story.router, prefix="/api", tags=["story"])

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.middleware("http")
async def cache_vendor(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/vendor/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/map-data", include_in_schema=False)
def map_data():
    return FileResponse(STATIC / "data" / "ph_regions.geojson", media_type="application/geo+json",
                        headers={"Cache-Control": "public, max-age=86400"})
