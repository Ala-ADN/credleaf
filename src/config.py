from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class ArchiveItCollection:
    id: int
    name: str
    credibility_tier: str  # "low" | "authoritative" | "mixed"


COVID_COLLECTIONS: dict[int, ArchiveItCollection] = {
    13559: ArchiveItCollection(
        id=13559,
        name="False Coronavirus (COVID-19) Information",
        credibility_tier="low",
    ),
    4887: ArchiveItCollection(
        id=4887,
        name="Global Health Events",
        credibility_tier="mixed",
    ),
    
    13529: ArchiveItCollection(
        id=13529,
        name="Novel Coronavirus (COVID-19)",
        credibility_tier="mixed",
    ),
}


@dataclass(frozen=True)
class Phase:
    name: str
    start: str  # ISO date
    end: str    # ISO date (inclusive)

    def cdx_window(self) -> tuple[str, str]:
        """Return (from_ts, to_ts) as 14-digit CDX timestamps."""
        f = datetime.fromisoformat(self.start).strftime("%Y%m%d000000")
        t = datetime.fromisoformat(self.end).strftime("%Y%m%d235959")
        return f, t


PANDEMIC_PHASES: list[Phase] = [
    # https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic

    # 2019-12-31 China reports cluster of pneumonia in Wuhan to WHO.
    Phase("P0_outbreak", "2019-12-01", "2020-01-30"),
    # 2020-01-30 WHO declares PHEIC.
    Phase("P1_global_onset", "2020-01-31", "2020-11-30"),
    # 2020-03-11 WHO characterizes COVID-19 as a pandemic.
    # 2020-12-02 UK authorizes Pfizer-BioNTech (world first).
    Phase("P2_pre_vaccine", "2020-12-01", "2020-12-31"),
    # 2021-01-01 broad rollout across high-income countries.
    Phase("P3_vaccine_rollout", "2021-01-01", "2021-05-31"),
    # 2021-05-11 WHO designates B.1.617.2 (Delta) a Variant of Concern.
    # 2021-06 Delta becomes globally dominant per WHO.
    Phase("P4_delta_wave", "2021-06-01", "2021-11-30"),
    # 2021-11-24 South Africa reports B.1.1.529 to WHO.
    # 2021-11-26 WHO designates Omicron a Variant of Concern.
    Phase("P5_omicron_wave", "2021-12-01", "2022-06-30"),
    # mid-2022 most countries lift emergency restrictions.
    Phase("P6_transition_end", "2022-07-01", "2023-05-05"),
    # 2023-05-05 WHO ends the COVID-19 PHEIC.
]

PHASES_BY_NAME: dict[str, Phase] = {p.name: p for p in PANDEMIC_PHASES}
