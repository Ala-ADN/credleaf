"""Canonical page identity shared by harvesting and downstream deduplication."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMETERS = frozenset({"fbclid", "gclid", "msclkid"})


def canonical_url(url: str) -> str:
    """Remove fragments/tracking parameters, retaining meaningful query values."""
    value = url.strip()
    parts = urlsplit(value if "://" in value else "https://" + value)
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return ""
    port = parts.port
    if ":" in host:
        host = f"[{host}]"
    if port and (parts.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMETERS]
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode(sorted(query)), ""))
