from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Collection(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    name: str
    publicly_visible: bool = True
    num_active_seeds: int = 0
    num_inactive_seeds: int = 0
    total_warc_bytes: int = 0
    created_date: datetime | None = None
    last_crawl_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Seed(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    collection: int
    url: str
    canonical_url: str | None = None
    seed_type: str | None = None
    active: bool = True
    publicly_visible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Capture(BaseModel):
    """One row of an Archive-It Wayback CDX response."""

    urlkey: str
    timestamp: str  # YYYYMMDDhhmmss
    url: str
    mimetype: str | None = None
    status: str | None = None
    digest: str | None = None
    redirect: str | None = None
    robotflags: str | None = None
    length: int | None = None
    offset: int | None = None
    filename: str | None = None

    @property
    def captured_at(self) -> datetime:
        return datetime.strptime(self.timestamp, "%Y%m%d%H%M%S")


class CaptureRecord(BaseModel):
    """A capture joined to its seed + collection context, ready for storage."""

    collection_id: int
    seed_id: int
    seed_url: str
    phase: str
    capture: Capture

    @property
    def wayback_url(self) -> str:
        return (
            f"https://wayback.archive-it.org/{self.collection_id}/"
            f"{self.capture.timestamp}id_/{self.capture.url}"
        )
