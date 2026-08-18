"""
STEP 1 - Fetch the page content behind every Source URL in the tracker.

What it does:
  - reads tracking_dataset_v4.xlsx (sheet "Dataset")
  - dedupes URLs (several rows share the same source)
  - skips domains that block scrapers (Crunchbase, LinkedIn - your row
    description already carries the useful info for those)
  - fetches HTML pages -> extracts clean article text with trafilatura
  - fetches PDFs -> extracts text with pymupdf
  - caches every raw download in ./cache so re-runs cost nothing
  - writes clean text to ./corpus/<id>.txt
  - writes fetch_manifest.csv mapping url -> file -> status

Safe to re-run any time: already-cached URLs are not downloaded again.
"""

import hashlib
import io
import time
import csv
from pathlib import Path

import pandas as pd
import requests
import trafilatura
import fitz  # pymupdf

XLSX = "tracking_dataset_v4.xlsx"
CACHE = Path("cache")
CORPUS = Path("corpus")
MANIFEST = "fetch_manifest.csv"

SKIP_DOMAINS = ("crunchbase.com", "linkedin.com", "facebook.com", "x.com", "twitter.com")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-tool/1.0"}
TIMEOUT = 25
DELAY = 1.0  # seconds between requests - be polite

CACHE.mkdir(exist_ok=True)
CORPUS.mkdir(exist_ok=True)


def url_id(url: str) -> str:
    """Stable short id for a URL, used as filename."""
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def fetch_raw(url: str) -> tuple[bytes | None, str]:
    """Download a URL (or load from cache). Returns (content, status)."""
    cached = CACHE / url_id(url)
    if cached.exists():
        return cached.read_bytes(), "cached"
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, f"http {r.status_code}"
        cached.write_bytes(r.content)
        time.sleep(DELAY)
        return r.content, "fetched"
    except requests.RequestException as e:
        return None, f"error: {type(e).__name__}"


def extract_text(url: str, content: bytes) -> str:
    """PDF -> pymupdf, HTML -> trafilatura."""
    is_pdf = url.lower().endswith(".pdf") or content[:5] == b"%PDF-"
    if is_pdf:
        with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    return trafilatura.extract(content) or ""


def main():
    df = pd.read_excel(XLSX, sheet_name="Dataset").fillna("")
    urls = sorted({u.strip() for u in df["Source URL"] if str(u).startswith("http")})
    print(f"{len(df)} rows, {len(urls)} unique URLs")

    rows = []
    for n, url in enumerate(urls, 1):
        uid = url_id(url)
        out_file = CORPUS / f"{uid}.txt"

        if any(d in url for d in SKIP_DOMAINS):
            rows.append({"url": url, "id": uid, "status": "skipped (blocked domain)", "chars": 0})
            continue
        if out_file.exists():
            rows.append({"url": url, "id": uid, "status": "done (previous run)",
                         "chars": out_file.stat().st_size})
            continue

        content, status = fetch_raw(url)
        if content is None:
            rows.append({"url": url, "id": uid, "status": status, "chars": 0})
            print(f"  [{n}/{len(urls)}] FAIL {status}  {url[:70]}")
            continue

        try:
            text = extract_text(url, content).strip()
        except Exception as e:
            rows.append({"url": url, "id": uid, "status": f"extract error: {type(e).__name__}", "chars": 0})
            continue

        if len(text) < 200:  # boilerplate-only or empty page
            rows.append({"url": url, "id": uid, "status": "too short / no readable text", "chars": len(text)})
            continue

        out_file.write_text(text, encoding="utf-8")
        rows.append({"url": url, "id": uid, "status": "ok", "chars": len(text)})
        if n % 20 == 0:
            print(f"  [{n}/{len(urls)}] fetched...")

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "id", "status", "chars"])
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"].startswith(("ok", "done")))
    print(f"\nDone. {ok}/{len(urls)} pages with usable text. Details in {MANIFEST}.")
    print("Failures are normal (dead links, paywalls, anti-bot). Move on - the")
    print("entity metadata (step 2) works regardless.")


if __name__ == "__main__":
    main()
