"""
STEP 1 - Fetch the page content behind every Source URL.

Deduplicates URLs, skips domains that block scrapers, extracts clean article
text with trafilatura and PDF text with pymupdf, caches every download so
re-runs cost nothing, and logs every outcome to fetch_manifest.csv.

  python 1_fetch_pages.py                  # skips URLs that failed before
  python 1_fetch_pages.py --retry-failed   # try the dead ones again

Failures are normal: dead academic domains, paywalls and anti-bot protection
account for roughly 20-30% of URLs. The manifest records which and why.

Successful fetches are cached, but failures are not, so without a memory of
past failures every run would spend twenty-plus seconds timing out on each
dead URL. The manifest is that memory.
"""

import csv
import hashlib
import sys
import io
import time
from pathlib import Path

import fitz  # pymupdf
import pandas as pd
import requests
import trafilatura

from pipeline_common import DATASET, SHEET

CACHE = Path("cache")
CORPUS = Path("corpus")
MANIFEST = "fetch_manifest.csv"
SKIP_DOMAINS = ("crunchbase.com", "linkedin.com", "facebook.com", "x.com", "twitter.com")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-tool/1.0"}
TIMEOUT = 25
DELAY = 1.0
RETRY_FAILED = "--retry-failed" in sys.argv

CACHE.mkdir(exist_ok=True)
CORPUS.mkdir(exist_ok=True)


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def fetch_raw(url: str):
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
    if url.lower().endswith(".pdf") or content[:5] == b"%PDF-":
        with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    return trafilatura.extract(content) or ""


def previously_failed() -> set:
    """URLs the last run could not retrieve. Skipped unless --retry-failed."""
    if RETRY_FAILED or not Path(MANIFEST).exists():
        return set()
    done = ("ok", "done", "cached", "fetched")
    with open(MANIFEST, encoding="utf-8") as f:
        return {r["url"] for r in csv.DictReader(f)
                if not r["status"].startswith(done)}


def main():
    df = pd.read_excel(DATASET, sheet_name=SHEET).fillna("")
    dead = previously_failed()
    if dead:
        print(f"{len(dead)} URLs failed on a previous run and will be skipped. "
              f"Use --retry-failed to try them again.")
    urls = sorted({u.strip() for u in df["Source URL"] if str(u).startswith("http")})
    print(f"{len(df)} records, {len(urls)} unique URLs")

    rows = []
    for n, url in enumerate(urls, 1):
        uid = url_id(url)
        out_file = CORPUS / f"{uid}.txt"

        if any(d in url for d in SKIP_DOMAINS):
            rows.append({"url": url, "id": uid, "status": "skipped (blocked domain)", "chars": 0})
            continue
        if url in dead and not out_file.exists():
            rows.append({"url": url, "id": uid, "status": "failed previously (skipped)",
                         "chars": 0})
            continue
        if out_file.exists():
            rows.append({"url": url, "id": uid, "status": "done (previous run)",
                         "chars": out_file.stat().st_size})
            continue

        content, status = fetch_raw(url)
        if content is None:
            rows.append({"url": url, "id": uid, "status": status, "chars": 0})
            print(f"  [{n}/{len(urls)}] FAIL {status}  {url[:68]}")
            continue
        try:
            text = extract_text(url, content).strip()
        except Exception as e:
            rows.append({"url": url, "id": uid,
                         "status": f"extract error: {type(e).__name__}", "chars": 0})
            continue
        if len(text) < 200:
            rows.append({"url": url, "id": uid, "status": "too short / no readable text",
                         "chars": len(text)})
            continue

        out_file.write_text(text, encoding="utf-8")
        rows.append({"url": url, "id": uid, "status": "ok", "chars": len(text)})
        if n % 20 == 0:
            print(f"  [{n}/{len(urls)}] ...")

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "id", "status", "chars"])
        w.writeheader(); w.writerows(rows)

    ok = sum(1 for r in rows if r["status"].startswith(("ok", "done")))
    print(f"\n{ok}/{len(urls)} pages with usable text. Details in {MANIFEST}.")


if __name__ == "__main__":
    main()
