"""Toponym recognition + GeoNames lookup → per-article geographic metadata.

Pipeline mirrors `normalize` and `embed`:
    normalized JSONL → ToponymExtractor → Gazetteer → GeoDoc JSONL
"""
from .extract import ToponymExtractor
from .gazetteer import Gazetteer
from .georef import GeorefStats, georef_phase
from .models import GeoDoc, GeoEntry, GeoMention, RawMention

__all__ = [
    "GeoDoc",
    "GeoEntry",
    "GeoMention",
    "Gazetteer",
    "GeorefStats",
    "RawMention",
    "ToponymExtractor",
    "georef_phase",
]
