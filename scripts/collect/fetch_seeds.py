"""Cloud fetch leg for router collection seeds (runs in GitHub Actions).

Generic only: a seed is a URL plus an optional link pattern for listing pages.
No per-site code. Politeness is global: 1 request/second per host, retry on
429/503 with backoff, browser UA with a contact string, Accept: application/pdf.

Outputs (into --out): results.jsonl (one record per attempt) and fetched/
containing every downloaded PDF named by its sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "(+cerebrum-slm-collector; GitHub bopoadz-del/cerebrum-slm)"
)

DEFAULT_SEEDS = [
    # 6x NRC ADAMS (previously 403 from the laptop)
    "https://www.nrc.gov/docs/ML0728/ML072880521.pdf",
    "https://www.nrc.gov/docs/ML0723/ML072320421.pdf",
    "https://www.nrc.gov/docs/ML0723/ML072320413.pdf",
    "https://www.nrc.gov/docs/ML1522/ML15229A301.pdf",
    "https://www.nrc.gov/docs/ML1414/ML14148A113.pdf",
    "https://www.nrc.gov/docs/ML2205/ML22056A509.pdf",
    # 1x BoardDocs (previously bot-walled)
    "https://go.boarddocs.com/mo/ladue/Board.nsf/files/CYAR2Q6B565D/$file/2.%20Performance%20Services%20Pay%20App..pdf",
    # Ogunquit OAC binders (town DocumentCenter; one project; up to 3 binders)
    "https://ogunquit.gov/DocumentCenter/View/4357/OAC-10-Binder",
    "https://ogunquit.gov/DocumentCenter/View/4361/OAC-11-Binder",
    "https://ogunquit.gov/DocumentCenter/View/4358/OAC-8-Binder",
]
# Listing pages: fetch HTML, then every href matching the pattern (generic).
DEFAULT_LISTINGS = [
    {
        "url": "https://ncsx.pppl.gov/NCSX_Construction/MCWF_Contracts/Quality_Review/Open_NCR-CA-RFD/",
        "pattern": r"NCR[^\"']*\.pdf",
        "class": "NCR",
    },
]


@dataclass
class Result:
    url: str
    status: int | None = None
    ok: bool = False
    sha256: str | None = None
    chars: int = 0
    title: str = ""
    error: str = ""
    fetched_at: str = ""
    class_: str = ""

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "status": self.status,
            "ok": self.ok,
            "sha256": self.sha256,
            "text_chars": self.chars,
            "page_title": self.title,
            "error": self.error,
            "fetch_timestamp": self.fetched_at,
            "class": self.class_,
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _title(html: bytes) -> str:
    m = re.search(rb"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if not m:
        return ""
    return m.group(1).decode("utf-8", "replace").strip()[:200]


class Fetcher:
    def __init__(self, out: Path, rate: float = 1.0, retries: int = 3) -> None:
        self.out = out
        self.rate = rate
        self.retries = retries
        self._last: dict[str, float] = {}
        (out / "fetched").mkdir(parents=True, exist_ok=True)

    def _politeness(self, host: str) -> None:
        now = time.monotonic()
        last = self._last.get(host, 0.0)
        wait = (1.0 / self.rate) - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def fetch(self, url: str, class_: str = "") -> Result:
        res = Result(url=url, class_=class_, fetched_at=_now())
        host = urllib.request.urlparse(url).netloc
        last_err = ""
        for attempt in range(1, self.retries + 1):
            self._politeness(host)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": UA,
                    "Accept": "application/pdf, text/html;q=0.9, */*;q=0.8",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                    res.status = r.status
                    if data[:4] == b"%PDF":
                        res.sha256 = hashlib.sha256(data).hexdigest()
                        # Text-layer quality is verified locally (fitz); the cloud leg
                        # only proves the fetch. A 200 + PDF magic is ok here.
                        res.ok = True
                        name = f"{res.sha256[:24]}.pdf"
                        (self.out / "fetched" / name).write_bytes(data)
                    else:
                        res.title = _title(data[:200000])
                        res.error = "not a PDF (listing/html)"
                    return res
            except urllib.error.HTTPError as e:
                res.status = e.code
                last_err = f"HTTP {e.code}"
                if e.code in (429, 503) and attempt < self.retries:
                    time.sleep(5 * attempt)
                    continue
            except Exception as e:  # noqa: BLE001
                last_err = type(e).__name__
                if attempt < self.retries:
                    time.sleep(5 * attempt)
                    continue
        res.error = last_err
        return res

    def listing(self, url: str, pattern: str, class_: str) -> list[Result]:
        """Fetch an HTML listing page and return results for every matched link."""
        self._politeness(urllib.request.urlparse(url).netloc)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                html = r.read()
        except Exception as e:  # noqa: BLE001
            return [Result(url=url, error=f"listing failed: {type(e).__name__}", class_=class_)]
        links = sorted({m.group(0) for m in re.finditer(pattern, html.decode("utf-8", "replace"))})
        base = url.rsplit("/", 1)[0] + "/"
        out: list[Result] = []
        for link in links:
            target = link if link.startswith("http") else base + link
            out.append(self.fetch(target, class_))
        if not links:
            out.append(Result(url=url, error="listing matched no links", class_=class_))
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("./out"))
    ap.add_argument("--seeds", type=Path, help="jsonl of {url, class} seeds (default seed list)")
    ap.add_argument("--listings", type=Path, help="jsonl of {url, pattern, class} listing pages")
    args = ap.parse_args(argv)

    seeds = [{"url": u, "class": ""} for u in DEFAULT_SEEDS]
    listings = DEFAULT_LISTINGS
    if args.seeds:
        seeds = [json.loads(l) for l in args.seeds.read_text(encoding="utf-8").splitlines() if l.strip()]
        listings = []  # a search-driven run re-fetches nothing from the default lists
    if args.listings:
        listings = [json.loads(l) for l in args.listings.read_text(encoding="utf-8").splitlines() if l.strip()]

    f = Fetcher(args.out)
    results: list[Result] = []
    for seed in seeds:
        results.append(f.fetch(seed["url"], seed.get("class", "")))
    for spec in listings:
        results.extend(f.listing(spec["url"], spec["pattern"], spec.get("class", "")))

    (args.out / "results.jsonl").write_text(
        "".join(json.dumps(r.as_dict()) + "\n" for r in results), encoding="utf-8"
    )
    ok = sum(1 for r in results if r.ok)
    print(f"done: {len(results)} attempts, {ok} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
