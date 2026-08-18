# AI Tracker RAG — clean rebuild

Pipeline: **xlsx → (fetch pages) → Qdrant (2 collections) → Kimi answers with filters + sources.**

## Folder layout (put everything in one folder, e.g. `rag_v2`)

```
rag_v2/
├── tracking_dataset_v4.xlsx    <- the dataset (single source of truth)
├── requirements.txt
├── .env                        <- your key (copy .env.example, fill it in)
├── 01_fetch_pages.py
├── 02_ingest_entities.py
├── 03_ingest_pages.py
├── 04_query.py
├── cache/                      <- created automatically (raw downloads)
└── corpus/                     <- created automatically (clean text)
```

## One-time setup

```bash
# 1. fresh virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# 2. install dependencies
pip install -r requirements.txt

# 3. API key: copy .env.example to .env, paste your Moonshot key.
#    Then DELETE key.txt. A bare txt file with an API key is how keys leak.

# 4. start Qdrant (Docker Desktop must be running)
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

The old `books_demo` and `ai_tracker` collections are dead weight now — either
ignore them or delete them from the Qdrant dashboard (http://localhost:6333/dashboard).

## Run order

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python 01_fetch_pages.py` | Downloads + cleans the ~360 source pages (HTML via trafilatura, PDFs via pymupdf). Caches everything, skips Crunchbase/LinkedIn. | 10–20 min, re-runs are instant |
| 2 | `python 02_ingest_entities.py` | 376 entities → collection `ai_entities`, every column as filterable payload + indexes. | ~5 min on CPU |
| 3 | `python 03_ingest_pages.py` | Chunks the fetched pages → collection `ai_pages`. Each chunk inherits its row's country/sector metadata. | 15–30 min on CPU |
| 4 | `python 04_query.py` | Ask questions. Searches both collections, Kimi answers with sources. | interactive |

Steps 1 and 3 are optional-but-recommended: step 2 + 4 alone already give you
a working filtered tracker Q&A. Add the page layer when you have time.

## Using the query tool

```
Question: country=Lebanon What does the national AI strategy target?
Question: sector=Healthcare Which startups do medical imaging?
Question: country=Morocco sector=Finance Who invests in AI startups?
Question: Compare Arabic NLP work in Tunisia and Egypt
```

Sector values: Education, Healthcare, Agriculture, Energy, Industry,
Maritime, Finance, Government, Cross-sector.

## Expected reality checks

- **Step 1 will show failures.** Dead links, paywalls, anti-bot pages — 20–30%
  loss is normal. The manifest (`fetch_manifest.csv`) tells you exactly which
  URLs failed and why. The tool works fine without them.
- **First run of any script downloads bge-m3 (~2 GB), once.**
- If `kimi-k2.6` errors, check the model name in the Moonshot console —
  they rename models.
- Re-running any ingest script is safe: collections are rebuilt from scratch.

## What changed vs your old files

- Scripts read the **xlsx directly** — your `urls_with_metadata.csv` was stale
  (still had Libya, missing the Application Sector column). Delete it.
- **Two collections, one query tool**: structured entities + page content,
  searched together. Your old `query.py`/`query2.py` (books demo) are retired.
- **Filters actually exist now** (`country=`, `sector=`), using the payload
  indexes — this was the "no filters yet" TODO in your old `query_dataset.py`.
- Payload keys are clean snake_case (`application_sector`, not
  `"Application Sector"`), so filter code stays readable.
- Temperature 0.3 instead of 1 — factual Q&A shouldn't be creative.
