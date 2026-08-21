"""User-behavior event collection for the observability dashboard.

The frontend pings this endpoint for lightweight funnel events (AQI viewed,
Why drawer opened, briefing completed). Events land in ``user_events`` and
roll up into the Scale section of the metrics report. Endpoint names are
restricted to [a-z0-9_] so the table can't be spammed with arbitrary rows.
"""

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.metrics import record_user_event

router = APIRouter(prefix="/api", tags=["Events"])

_EVENT_NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")


class UserEvent(BaseModel):
    event: str
    detail: Optional[str] = None


@router.post("/events")
async def post_event(ev: UserEvent):
    name = (ev.event or "").strip()
    if not _EVENT_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="event must match [a-z0-9_] (max 64 chars)")
    record_user_event(name, ev.detail)
    return {"ok": True}
