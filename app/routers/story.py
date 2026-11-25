"""Plain-language story of the inflation picture, written from the data (no hand-typed numbers)."""
from __future__ import annotations

from datetime import date as _date

from fastapi import APIRouter, Query

from app import db
from app.services import models

router = APIRouter()

BAND_HI = 4.0
GLOSSARY = [
    {"term": "Inflation", "plain": "How much more the same basket of goods costs now than a year ago."},
    {"term": "CPI", "plain": "Consumer Price Index: the price tag of a typical household basket. Inflation is its change over a year."},
    {"term": "BSP target band (2-4%)", "plain": "The range the central bank aims to keep inflation inside."},
    {"term": "Core inflation", "plain": "Inflation without the jumpy items (food and fuel). Shows the underlying trend."},
    {"term": "Forecast band", "plain": "The shaded range where we expect inflation to land. The wider the band, the less certain we are."},
    {"term": "Carry-forward", "plain": "A simple guess that next month looks like this month. It is hard to beat, so we show it beside every model."},
    {"term": "Breach chance", "plain": "How likely a region is to be above 4% next month, according to the model."},
]


def _month(d) -> str:
    return _date.fromisoformat(str(d)[:10]).strftime("%B %Y")


def _pp(x: float) -> str:
    return f"{x:.1f}%"


def _streak(rows: list[dict]) -> int:
    n = 0
    for r in reversed(rows):
        if r["headline"] is not None and r["headline"] > BAND_HI:
            n += 1
        else:
            break
    return n


def _first_above(rows: list[dict]) -> str | None:
    """Start of the current run above the band."""
    k = _streak(rows)
    return str(rows[-k]["date"])[:10] if k else None


@router.get("/story")
def story(end: str = Query("2026-10-01")):
    kp = db.query("kpis", [end]).rows[0]
    hist = db.query("headline", ["2018-01-01", end]).rows
    now = kp["headline"]
    prev = kp["prev"]
    when = _month(kp["date"])
    streak = _streak(hist)
    last24 = hist[-24:]
    low = min(last24, key=lambda r: r["headline"])
    chapters = []

    # 1. where are we
    status = "above" if now > BAND_HI else ("below" if now < 2.0 else "inside")
    direction = "up from" if prev is not None and now > prev else "down from"
    b1 = [f"In {when}, prices were {_pp(now)} higher than a year earlier, {direction} {_pp(prev)} the month before."]
    if status == "above":
        b1.append(f"That is above the central bank's 2-4% comfort range"
                  + (f", and it has been above for {streak} months in a row." if streak > 1 else "."))
        before = hist[-streak - 1] if 0 < streak < len(hist) else None
        if before is not None:
            b1.append(f"As recently as {_month(before['date'])} it was {_pp(before['headline'])}, so this is a recent jump, not a long-standing problem.")
    elif status == "inside":
        b1.append("That is inside the central bank's 2-4% comfort range.")
    else:
        b1.append("That is below the central bank's 2-4% comfort range.")
    if kp.get("core") is not None:
        b1.append(f"Even after removing food and fuel (core inflation), prices are rising {_pp(kp['core'])}, "
                  + ("so the pressure is broad, not only from a few items." if kp["core"] > BAND_HI
                     else "so most of the pressure comes from food and fuel."))
    chapters.append({"id": "now", "title": "Where are we now?", "kicker": f"As of {when}",
                     "big": _pp(now), "bullets": [b for b in b1 if b],
                     "chart": "headline", "caption": "The line is how much higher prices are than a year earlier. The green band is the 2-4% target."})

    # 2. who is hit hardest
    mp = db.query("map", [end]).rows
    mp = [r for r in mp if r["value"] is not None]
    if mp:
        hi, lo = mp[0], mp[-1]
        def nm(r):
            head, _, tail = r["geo_id"].partition(" (")
            tail = tail.rstrip(")")
            return tail if tail.isupper() and len(tail) <= 6 else head
        above = sum(1 for r in mp if r["value"] > BAND_HI)
        b2 = [f"{nm(hi)} is highest at {_pp(hi['value'])}; {nm(lo)} is lowest at {_pp(lo['value'])}.",
              f"{above} of {len(mp)} regions are above the 4% ceiling.",
              f"The gap between the highest and lowest region is {hi['value'] - lo['value']:.1f} points: the same peso buys less in some places than others."]
        chapters.append({"id": "where", "title": "Who feels it most?", "kicker": "Regions compared",
                         "big": nm(hi), "bullets": b2, "chart": "map",
                         "caption": "Red bars are above the 4% ceiling. Click a region to focus the whole page on it."})

    # 3. drivers
    tm = [r for r in db.query("treemap", [end, "PHILIPPINES"]).rows if r["value"] is not None]
    if tm:
        pts = sorted(({"name": r["name"].title().split(",")[0], "pp": r["weight_pct"] * r["value"] / 100,
                       "value": r["value"], "weight": r["weight_pct"]} for r in tm), key=lambda r: -r["pp"])
        tot = sum(p["pp"] for p in pts)
        top3 = pts[:3]
        share = sum(p["pp"] for p in top3) / tot * 100 if tot else 0
        hot = max(pts, key=lambda p: p["value"])
        b3 = [f"About {share:.0f}% of today's inflation comes from just three groups: "
              + ", ".join(f"{p['name']} ({p['pp']:.1f} points)" for p in top3) + ".",
              f"The fastest riser is {hot['name']}, up {_pp(hot['value'])} in a year (it is {hot['weight']:.0f}% of a household budget)."]
        chapters.append({"id": "why", "title": "What is pushing prices up?", "kicker": "Drivers",
                         "big": top3[0]["name"].split(" And ")[0].replace("Food", "Food"), "bullets": b3, "chart": "treemap",
                         "caption": "Each bar is how many points a spending group adds to inflation. Red = prices in that group rising faster than 4%."})

    # 4. what next
    fc = models.FORECAST[models.FORECAST["model"] == "SARIMA"].sort_values("date") if not models.FORECAST.empty else None
    if fc is not None and not fc.empty:
        f1, f12 = fc.iloc[0], fc.iloc[min(11, len(fc) - 1)]
        b4 = [f"Our forecast for {_month(f1['date'])} is {_pp(f1['inflation'])}, most likely between {_pp(f1['infl_lo95'])} and {_pp(f1['infl_hi95'])}.",
              f"A year out ({_month(f12['date'])}) it is {_pp(f12['inflation'])}, but the range is far wider ({_pp(f12['infl_lo95'])} to {_pp(f12['infl_hi95'])}): the further ahead, the less sure we are."]
        ab = models.regions_expected_above(end)
        if ab.get("count") is not None:
            b4.append(f"For next month, the model expects {ab['count']} of {ab['total']} regions to stay above 4%.")
        chapters.append({"id": "next", "title": "What happens next?", "kicker": "Forecast",
                         "big": _pp(f1["inflation"]), "bullets": b4, "chart": "forecast",
                         "caption": "Dark line = best guess. Shaded bands = where we expect inflation to land (80% and 95% of the time)."})

    # 5. reliability (honest)
    m = models.METRICS
    nat = m.get("national_rolling", {}).get("mean", {})
    reg = {k: v for k, v in (m.get("nowcast", {}).get("folds", [{}])[-1].get("regression", {}) or {}).items()}
    persist = reg.get("Persistence (this month carried forward)", {}).get("MAE")
    deployed = reg.get("Boosted trees, change target + current rate (deployed)", {}).get("MAE")
    b5 = []
    if nat.get("SARIMA"):
        b5.append(f"Across four past test years the national forecast was off by about {nat['SARIMA']['MAE']:.1f} points on average, "
                  f"and its bands held the true value {nat['SARIMA'].get('coverage95', 0) * 100:.0f}% of the time (target 95%).")
        b5.append("It struggles in sudden price surges, when the bands were too narrow. Treat the forecast as a range, not a promise.")
    if persist is not None and deployed is not None:
        b5.append(f"For regions one month ahead, simply assuming 'next month looks like this month' (miss of {persist:.1f} points) beat our "
                  f"machine-learning model ({deployed:.1f}). So the headline regional guess is the carried-forward rate; the model adds the breach chance and explains what is moving it.")
    chapters.append({"id": "trust", "title": "How much should you trust this?", "kicker": "Honest limits",
                     "big": f"±{nat['SARIMA']['MAE']:.1f} pts" if nat.get("SARIMA") else "",
                     "bullets": b5, "chart": "backtest",
                     "caption": "Average miss of each approach in four past test periods (shorter bar = better)."})

    # 6. act
    chapters.append({"id": "act", "title": "What can be done?", "kicker": "Try it yourself",
                     "big": "What if?",
                     "bullets": ["Use the Forecast tab to ask 'what if fuel or rice prices jump 20%?' and see how the whole path moves.",
                                 "Use the Risk tab to test one region: change this month's rate and see how its breach chance responds.",
                                 "Local governments can use the regional view to target relief where inflation is highest; the central bank can compare against the 2-4% band."],
                     "chart": None, "caption": ""})
    events = []
    import json
    from app.config import settings
    f = settings.root / "data" / "news" / f"{str(kp['date'])[:10]}.json"
    if f.exists():
        try:
            events = [{"title": n["title"], "domain": n["domain"], "url": n["url"]}
                      for n in json.loads(f.read_text(encoding="utf-8"))[:3]]
        except (OSError, ValueError):
            events = []
    return {"data": {"as_of": str(kp["date"])[:10], "chapters": chapters, "glossary": GLOSSARY,
                     "news": events},
            "meta": {"panel_id": "story", "cache": "computed", "data_vintage": db.DATA_VINTAGE}}
