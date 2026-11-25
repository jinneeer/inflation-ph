from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    model: str = Field("sarima", pattern="^(sarima|ets|both)$")
    p: int = Field(1, ge=0, le=3)
    d: int = Field(1, ge=0, le=2)
    q: int = Field(1, ge=0, le=3)
    P: int = Field(1, ge=0, le=2)
    D: int = Field(1, ge=0, le=1)
    Q: int = Field(0, ge=0, le=2)
    trend: str = Field("add", pattern="^(add|none)$")
    seasonal: str = Field("add", pattern="^(add|none)$")
    damped: bool = True
    horizon: int = Field(24, ge=6, le=36)
    shock_node: str | None = None
    shock_pct: float = Field(0.0, ge=-60, le=60)
    shock_start: int = Field(1, ge=1, le=36)


class NowcastWhatIf(BaseModel):
    region: str
    infl: float = Field(..., ge=-10, le=30)


class ThresholdRequest(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
