"""URL normalisation. Two spellings of the same page must dedupe to one key."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"fbclid", "gclid", "ref", "ref_src", "mc_cid", "mc_eid"}


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Canonical form of ``url`` (resolved against ``base``), or None if not crawlable.

    * only http(s). mailto:, javascript:, data: and others are rejected
    * lowercase scheme and host, drop default ports and the ``#fragment``
    * drop tracking parameters (utm_*, fbclid...) and sort the rest
    * empty path becomes ``/``
    """
    if base is not None:
        url = urljoin(base, url.strip())
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower().rstrip(".")
    try:
        port = parts.port
    except ValueError:
        return None
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(_TRACKING_PREFIXES) and k.lower() not in _TRACKING_PARAMS
    ]
    query.sort()
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc


def in_domains(url: str, domains: set[str]) -> bool:
    """True if the URL's host is one of ``domains`` or a subdomain of one."""
    host = urlsplit(url).hostname or ""
    return any(host == d or host.endswith("." + d) for d in domains)
