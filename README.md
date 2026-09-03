# MED AI Tracker

A retrieval-augmented generation (RAG) system for querying the artificial
intelligence ecosystem of the Mediterranean, built on a manually verified
dataset of 637 records across eight countries. Every answer is assembled only
from catalogued records and cites its sources; nothing is answered from the
language model's own knowledge.

Built during a summer 2026 internship at the AI-Accelerated Research Centre
(AI-ARC), Mohammed VI Polytechnic University, Ben Guerir.

---

## What it does

Ask a question in plain language, get an answer grounded in verified records
with a source URL and verification status attached to every claim.

```
Which startups work on medical imaging?
Quelles startups marocaines utilisent l'IA dans l'agriculture ?
Compare the AI research capacity of Algeria and Egypt.
```

Questions can be asked in English, French or Arabic. Retrieval works across
languages: an English question retrieves French-language records describing the
same thing, because matching is by meaning rather than keyword.

---

## The dataset

`tracking_dataset.xlsx` — 637 records, eight countries.

| Southern shore | | Northern shore | |
|---|---|---|---|
| Egypt | 142 | France | 40 |
| Morocco | 113 | Italy | 30 |
| Tunisia | 111 | Spain | 30 |
| Algeria | 93 | | |
| Lebanon | 78 | | |

Covering startups, research laboratories, incubators and accelerators, funding
bodies, national policies and strategies, hackathons, conferences and
professional communities.

Each record carries a source URL, a source type (primary, government, news,
aggregator, secondary), an application sector, an entity type, a maturity
indicator, and a **verification status**:

- **Verified** — the cited source is primary or authoritative and confirms the claim
- **Not yet verified** — the claim rests on a single aggregated source
- **Mismatch found** — the source does not support the claim; excluded from the index

The verification status is the point of the dataset. A funding figure copied
from a startup directory and one confirmed against a company announcement are
not equally reliable, and the dataset records which is which. That status is
written into the text the model reads, so an unverified record is flagged in
the answer rather than presented with false confidence.

Current split: 565 verified, 71 not yet verified, 1 mismatch.

The `Change Log` sheet documents how the dataset was assembled, including the
merge of the five-country and northern-Mediterranean branches and every
taxonomy mapping applied.

---

## Setup

Requires Python 3.10+ and Docker.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

cp .env.example .env             # paste your Moonshot key into .env
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

The first run downloads the BGE-M3 embedding model (~2 GB), once. After that
everything runs locally on CPU at no per-query cost. Answer generation needs an
API key; retrieval does not.

Docker Desktop does not start automatically on Windows. A `WinError 10061`
means the container is not running: `docker start <container-id>`.

---

## Running it

Minimum to get a working system:

```bash
python 2_ingest_entities.py      # 637 records -> Qdrant, ~8 min on CPU
streamlit run app.py
```

Full pipeline, including the text of the source pages themselves:

```bash
python run_pipeline.py           # fetch pages -> ingest records -> ingest chunks
```

`run_pipeline.py` is incremental. Add rows to the spreadsheet and re-run: it
fetches only new URLs and embeds only records whose content changed. Adding
twenty records takes about a minute rather than half an hour. Records deleted
from the spreadsheet are removed from the database.

Use `--rebuild` only after changing the chunk-construction template or the
embedding model, since those invalidate every stored vector without changing
the source text.

Command line, without the dashboard:

```bash
python query.py "which startups work on medical imaging"
python query.py --country Egypt --sector Health "AI diagnostics"
python query.py --no-llm "AI in agriculture"     # retrieval only, no API key
```

---

## The dashboard

`streamlit run app.py` — six tabs:

- **Search** — filtered semantic search with a grounded, cited answer
- **Entity** — full record view, source link, nearest neighbours in vector space
- **Compare** — two countries side by side, including startup-to-research ratio
- **Map** — embedding projection, plus duplicate detection and candidate
  misclassifications
- **Overview** — composition, coverage matrix, thinnest cells
- **Validation** — evaluation results, manual grading, export for reports

The map's two analysis panels are quality-control tools rather than decoration.
One surfaces near-identical records that may be duplicates; the other flags
records whose description sits closer to another sector's centroid than to
their own label. Neither proves an error, but both narrow 637 records to the
handful worth re-reading.

---

## Validation

```bash
python evaluate.py
```

Runs a question set whose correct answers are declared in advance
(`eval_questions.json`) and reports:

- **Recall@5** on questions with known expected records
- **Score separation** between in-scope and deliberately out-of-scope
  questions. If out-of-scope questions scored as highly as in-scope ones, the
  system would have no signal for telling apart what it knows from what it
  does not.
- **A manual review queue** for properties string matching cannot check:
  refusal on unanswerable questions, cross-lingual retrieval, verification
  awareness, comparative answers. These are graded in the dashboard's
  Validation tab, saved to `eval_verdicts.json`, and folded into the headline
  figures. The tab exports a CSV, a LaTeX table and a summary paragraph.

The most informative test asks for a figure that does not exist about an entity
that does. A system that invents a number there fails in the way that matters
most.

---

## How it works

```
OFFLINE   spreadsheet -> prose passage per record -> BGE-M3 -> Qdrant
          source URLs -> fetched text -> chunks -> Qdrant (metadata inherited)

ONLINE    question -> embedding -> filtered vector search -> LLM -> cited answer
```

Three design decisions worth knowing:

**Records are embedded as prose, not field values.** A record becomes *"FarmAI
is a startup based in Algeria, working in Agriculture (Computer Vision).
[description]. It has raised $100,000 in reported funding."* Questions are
asked in prose, and passages shaped like the expected question retrieve better
than comma-separated fields. Verification caveats are appended to the passage
before embedding, so they reach the model automatically.

**Metadata filtering happens in the database.** Entities doing similar work in
different countries produce similar vectors, so semantic similarity alone does
not respect country boundaries. Country and sector filters are indexed Qdrant
payload conditions, not post-processing.

**Retrieval is diversity-capped.** The system over-fetches, then admits at most
two passages per source URL. Without this, one densely matching source occupies
every slot and crowds out other relevant entities.

---

## Limitations

- 71 records rest on a single aggregated source and are marked *Not yet
  verified*. They are indexed, but flagged wherever they appear.
- 116 records merged from the northern-Mediterranean branch have no maturity
  value, because that branch did not record one. It was left blank rather than
  inferred.
- The dataset is a periodic snapshot with no automatic refresh. Funding figures
  and organisational status go stale.
- Maritime (14 records) and energy (22) are too thinly covered to support
  conclusions.
- Retrieval parameters were tuned against the same question set used to
  evaluate them, so reported recall is optimistic.
- Source-page fetching fails for roughly 20-30% of URLs: dead academic domains,
  paywalls, anti-bot protection. Failures are logged in `fetch_manifest.csv`.

---

## Repository layout

```
tracking_dataset.xlsx     the dataset, with summary, charts and change log
pipeline_common.py        shared config and incremental-ingestion helpers
run_pipeline.py           runs the three steps below in order
  1_fetch_pages.py          download and extract source page text
  2_ingest_entities.py      records -> Qdrant (incremental)
  3_ingest_pages.py         page chunks -> Qdrant (incremental)
query.py                  command line; --no-llm for retrieval only
evaluate.py               validation harness
eval_questions.json       evaluation set with expected answers
app.py                    Streamlit dashboard
```

The dataset filename and collection names are defined once in
`pipeline_common.py`. Change them there and every script follows.

---

## Adding records

1. Add rows to the spreadsheet, filling every column. A source URL is mandatory.
2. Set the verification status honestly. If the only source is a startup
   directory or a listicle, it is *Not yet verified*, however plausible it looks.
3. Run `python run_pipeline.py`. Only what changed is processed.
4. Run `python evaluate.py` to confirm nothing regressed.

Blank maturity values, missing URLs and duplicate country+name pairs are
reported in `dataset_quality_report.txt` after every ingestion run.
