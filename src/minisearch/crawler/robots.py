"""robots.txt handling, cached per host.

Follows RFC 9309:

* 2xx: parse the rules.
* 4xx (e.g. 404): there are no rules, so everything is allowed.
* 5xx / network error: assume complete disallow. An unreachable robots.txt
  must not be read as "anything goes".

``Crawl-delay`` is honoured. The effective delay for a host is
``max(configured delay, Crawl-delay)``. ``urllib.robotparser`` accepts only
*integer* delays, silently ignoring ``Crawl-delay: 0.5``, so the delay is
parsed here (floats; our own user-agent's group beats ``*``). Allow/Disallow
matching uses the standard library's ``urllib.robotparser``. Only the core search engine has to be
written from scratch, and re-implementing a parser for a public spec would
add risk without adding anything.
"""

from __future__ import annotations

from urllib.robotparser import RobotFileParser

import httpx

from minisearch.crawler.urls import host_of


def parse_crawl_delay(text: str, user_agent: str) -> float | None:
    """Crawl-delay for ``user_agent`` (its own group first, then ``*``)."""
    token = user_agent.split("/")[0].strip().lower()
    groups: dict[str, float] = {}
    agents: list[str] = []
    in_rules = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (x.strip() for x in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if in_rules:  # a new group starts
                agents, in_rules = [], False
            agents.append(value.lower())
        else:
            in_rules = True
            if key == "crawl-delay":
                try:
                    delay = float(value)
                except ValueError:
                    continue
                for a in agents:
                    groups[a] = delay
    for agent, delay in groups.items():
        if agent != "*" and agent in token:
            return delay
    return groups.get("*")


class RobotsCache:
    def __init__(self, client: httpx.Client, user_agent: str) -> None:
        self.client = client
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser | bool] = {}
        self._delays: dict[str, float | None] = {}

    def _load(self, url: str) -> RobotFileParser | bool:
        host = host_of(url)
        if host in self._cache:
            return self._cache[host]
        scheme = url.split(":", 1)[0]
        robots_url = f"{scheme}://{host}/robots.txt"
        result: RobotFileParser | bool
        try:
            resp = self.client.get(robots_url, timeout=10.0)
            if 200 <= resp.status_code < 300:
                rp = RobotFileParser(robots_url)
                rp.parse(resp.text.splitlines())
                self._delays[host] = parse_crawl_delay(resp.text, self.user_agent)
                result = rp
            elif 400 <= resp.status_code < 500:
                result = True  # no robots.txt: allow all
            else:
                result = False  # server error: disallow all
        except httpx.HTTPError:
            result = False
        self._cache[host] = result
        return result

    def allowed(self, url: str) -> bool:
        rules = self._load(url)
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        self._load(url)
        return self._delays.get(host_of(url))
