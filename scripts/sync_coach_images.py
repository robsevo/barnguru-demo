"""sync_coach_images — Populate ``image_url`` for every head coach in data/coaches.json.

Queries the Wikipedia REST API summary endpoint for each coach and writes
the ``thumbnail`` URL (or ``originalimage`` as a fallback) back to
``data/coaches.json``. Idempotent: existing ``image_url`` values are kept
unless ``--force`` is passed.

The team page (``/teams/{abbrev}``) and coach directory page (``/coaches``)
both already render ``image_url`` with an HC fallback, so this is the
data-side of the pipeline.

Usage::

    uv run python scripts/gretzky.py sync-coach-images
    uv run python scripts/gretzky.py sync-coach-images -- --force
    uv run python scripts/gretzky.py sync-coach-images -- --only "Jon Cooper"

Per the Real Data Policy: every coach we couldn't resolve gets printed
loudly. The script exits non-zero if any required coach is still missing
an image_url at the end (so CI / nightly catches roster drift).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

COACHES_PATH = _REPO / "data" / "coaches.json"

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
USER_AGENT = "GRTZKY/1.0 (http://localhost:3000; coach-image-sync) python-httpx/0.27"

# Coach names that need explicit Wikipedia disambiguation. Wikipedia
# sometimes has multiple people with the same name; the unqualified
# page either disambiguates or points at the wrong person. Override
# here with the exact Wikipedia article title we want.
WIKI_OVERRIDES: dict[str, str] = {
    "Greg Cronin":       "Greg Cronin",
    "Joe Sacco":         "Joe Sacco (ice hockey)",
    "Lindy Ruff":        "Lindy Ruff",
    "Ryan Huska":        "Ryan Huska",
    "Rod Brind'Amour":   "Rod Brind'Amour",
    "Anders Sorensen":   "Anders Sörensen",
    "Jared Bednar":      "Jared Bednar",
    "Dean Evason":       "Dean Evason",
    "Pete DeBoer":       "Peter DeBoer",
    "Todd McLellan":     "Todd McLellan",
    "Kris Knoblauch":    "Kris Knoblauch",
    "Paul Maurice":      "Paul Maurice",
    "Jim Hiller":        "Jim Hiller",
    "John Hynes":        "John Hynes (ice hockey)",
    "Martin St. Louis":  "Martin St. Louis",
    "Andrew Brunette":   "Andrew Brunette",
    "Sheldon Keefe":     "Sheldon Keefe",
    "Patrick Roy":       "Patrick Roy",
    "Mike Sullivan":     "Mike Sullivan (ice hockey)",
    "Travis Green":      "Travis Green",
    "Rick Tocchet":      "Rick Tocchet",
    "Dan Muse":          "Dan Muse",
    "Ryan Warsofsky":    "Ryan Warsofsky",
    "Lane Lambert":      "Lane Lambert",
    "Jim Montgomery":    "Jim Montgomery (ice hockey)",
    "Jon Cooper":        "Jon Cooper (ice hockey)",
    "Craig Berube":      "Craig Berube",
    "André Tourigny":    "André Tourigny",
    "Adam Foote":        "Adam Foote",
    "Bruce Cassidy":     "Bruce Cassidy",
    "Spencer Carbery":   "Spencer Carbery",
    "Scott Arniel":      "Scott Arniel",
}


def _wiki_title(name: str) -> str:
    """Return the Wikipedia article title to query for this coach."""
    return WIKI_OVERRIDES.get(name, name)


def _slug(title: str) -> str:
    """Wikipedia REST expects underscores, percent-encoded specials."""
    return title.strip().replace(" ", "_")


def fetch_image_for(client: httpx.Client, name: str) -> tuple[str | None, str]:
    """Return (image_url, status_msg) for a coach.

    image_url is None on any failure. status_msg describes what happened
    (always printed) so we can audit the run.
    """
    title = _wiki_title(name)
    slug = _slug(title)
    url = WIKI_SUMMARY.format(slug=slug)
    try:
        r = client.get(url, timeout=15.0)
    except httpx.HTTPError as e:
        return None, f"network error: {e!r}"

    if r.status_code == 404:
        return None, f"404 for title='{title}'"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code} for title='{title}'"

    try:
        payload: dict[str, Any] = r.json()
    except json.JSONDecodeError as e:
        return None, f"non-JSON response: {e!r}"

    if payload.get("type") == "disambiguation":
        return None, f"disambiguation page for title='{title}' (add override)"

    thumb = (payload.get("thumbnail") or {}).get("source")
    full = (payload.get("originalimage") or {}).get("source")
    img = thumb or full
    if not img:
        return None, f"no image on page title='{title}'"

    return img, f"ok title='{title}'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even when image_url is already set.")
    parser.add_argument("--only", default=None,
                        help="Only sync this one coach (exact name match).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results but don't write coaches.json.")
    args = parser.parse_args()

    if not COACHES_PATH.exists():
        print(f"[coach-images] FATAL: {COACHES_PATH} does not exist", file=sys.stderr)
        sys.exit(2)

    blob = json.loads(COACHES_PATH.read_text(encoding="utf-8"))
    coaches = blob.get("coaches") or []
    if not isinstance(coaches, list) or not coaches:
        print(f"[coach-images] FATAL: {COACHES_PATH} has no coaches[] list", file=sys.stderr)
        sys.exit(2)

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    n_ok = 0
    n_skip = 0
    n_fail = 0
    failures: list[str] = []

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for c in coaches:
            name = c.get("name")
            if not name:
                continue
            if args.only and name != args.only:
                continue

            existing = c.get("image_url")
            if existing and not args.force:
                print(f"[coach-images] SKIP  {name:24s}  (already has image_url)")
                n_skip += 1
                continue

            img, status = fetch_image_for(client, name)
            if img:
                c["image_url"] = img
                print(f"[coach-images] OK    {name:24s}  {status}  -> {img}")
                n_ok += 1
            else:
                print(f"[coach-images] FAIL  {name:24s}  {status}", file=sys.stderr)
                failures.append(f"{name}: {status}")
                n_fail += 1

            # Wikipedia's REST API is generous but be a polite citizen.
            time.sleep(0.2)

    if args.dry_run:
        print(f"[coach-images] DRY-RUN — would write {n_ok} new urls; skipped {n_skip}; failed {n_fail}")
    else:
        COACHES_PATH.write_text(
            json.dumps(blob, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[coach-images] wrote {COACHES_PATH}  (ok={n_ok} skip={n_skip} fail={n_fail})")

    if n_fail:
        print(f"[coach-images] {n_fail} coaches still missing image_url:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        # Non-zero exit so nightly catches regressions
        sys.exit(1)


if __name__ == "__main__":
    main()
