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
        #CONTAINS NON COVID CONTENT
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
    #Phase("P0_outbreak", "2019-12-01", "2020-01-30"),
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


@dataclass(frozen=True)
class PandemicEvent:
    """A dated historical milestone, independent of the corpus sampling phases."""

    date: str  # ISO date of the event (not the publication date of its source)
    title: str  # short enough for an animation callout
    description: str
    source_url: str


# Curated from Wikipedia's pandemic timeline and the linked response articles.
# Keep these in chronological order. Store the facts locally so notebook reruns
# are reproducible and do not require live Wikipedia requests.
COVID_TIMELINE_EVENTS: tuple[PandemicEvent, ...] = (
    PandemicEvent(
        "2019-12-31", "Wuhan reports pneumonia cluster",
        "Wuhan health officials publish a notice about unexplained pneumonia.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_2019#31_December",
    ),
    PandemicEvent(
        "2020-01-23", "Wuhan enters lockdown",
        "Travel restrictions begin in Wuhan as the outbreak grows.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_January_2020#23_January",
    ),
    PandemicEvent(
        "2020-01-30", "WHO declares global emergency",
        "WHO declares a Public Health Emergency of International Concern.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_January_2020#30_January",
    ),
    PandemicEvent(
        "2020-03-09", "Italy announces national lockdown",
        "Italy extends movement restrictions across the country.",
        "https://en.wikipedia.org/wiki/COVID-19_lockdowns_in_Italy",
    ),
    PandemicEvent(
        "2020-03-11", "WHO characterizes a pandemic",
        "WHO describes the worldwide COVID-19 outbreak as a pandemic.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_March_2020#11_March",
    ),
    PandemicEvent(
        "2020-03-24", "India announces national lockdown",
        "A 21-day nationwide lockdown is announced, starting on 25 March.",
        "https://en.wikipedia.org/wiki/Responses_to_the_COVID-19_pandemic_in_March_2020#24_March_2",
    ),
    PandemicEvent(
        "2020-04-02", "One million confirmed cases",
        "Reported COVID-19 cases worldwide pass one million.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_April_2020#2_April",
    ),
    PandemicEvent(
        "2020-12-02", "UK authorizes Pfizer vaccine",
        "The UK authorizes the Pfizer-BioNTech vaccine for mass immunisation.",
        "https://en.wikipedia.org/wiki/Responses_to_the_COVID-19_pandemic_in_December_2020#2_December",
    ),
    PandemicEvent(
        "2020-12-08", "UK begins Pfizer vaccinations",
        "The UK begins its public Pfizer-BioNTech vaccination programme.",
        "https://en.wikipedia.org/wiki/Responses_to_the_COVID-19_pandemic_in_December_2020#8_December",
    ),
    PandemicEvent(
        "2021-05-31", "WHO introduces the Delta name",
        "WHO assigns Greek-letter names to variants, including Delta.",
        "https://en.wikipedia.org/wiki/SARS-CoV-2_Delta_variant",
    ),
    PandemicEvent(
        "2021-11-26", "Omicron designated a concern",
        "WHO names Omicron and classifies it as a variant of concern.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_November_2021#26_November",
    ),
    PandemicEvent(
        "2022-03-28", "Shanghai lockdown begins",
        "A phased lockdown begins in Shanghai during an Omicron outbreak.",
        "https://en.wikipedia.org/wiki/2022_Shanghai_COVID-19_outbreak",
    ),
    PandemicEvent(
        "2022-12-07", "China eases zero-COVID rules",
        "China rolls back major testing and quarantine restrictions.",
        "https://en.wikipedia.org/wiki/Zero-COVID#China",
    ),
    PandemicEvent(
        "2023-05-05", "WHO ends global emergency",
        "WHO ends the emergency designation; COVID-19 remains a health threat.",
        "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_2023#5_May",
    ),
)
