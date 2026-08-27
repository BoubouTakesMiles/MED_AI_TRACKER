"""
Run the whole pipeline in order. This is the "I added rows, update everything"
button.

  python run_pipeline.py              # incremental: only what changed
  python run_pipeline.py --rebuild    # force full re-embed (steps 2 and 3)
  python run_pipeline.py --skip-fetch # don't hit the network at all

Each step is safe to re-run. Fetching skips cached URLs; ingestion skips
unchanged content. Adding 20 rows to the spreadsheet costs about a minute
instead of half an hour.
"""

import subprocess
import sys
import time

REBUILD = "--rebuild" in sys.argv
SKIP_FETCH = "--skip-fetch" in sys.argv

steps = []
if not SKIP_FETCH:
    steps.append(("Fetch source pages", ["01_fetch_pages.py"]))
steps += [
    ("Ingest entities", ["02_ingest_entities.py"] + (["--rebuild"] if REBUILD else [])),
    ("Ingest page chunks", ["03_ingest_pages.py"] + (["--rebuild"] if REBUILD else [])),
]

t0 = time.time()
for i, (label, cmd) in enumerate(steps, 1):
    print(f"\n{'=' * 66}\n[{i}/{len(steps)}] {label}\n{'=' * 66}")
    started = time.time()
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\nStopped: '{label}' failed with exit code {result.returncode}.")
        print("Fix the error above and re-run - completed steps won't redo their work.")
        sys.exit(result.returncode)
    print(f"[{label} finished in {time.time() - started:.0f}s]")

print(f"\n{'=' * 66}")
print(f"Pipeline complete in {time.time() - t0:.0f}s.")
print("Query with:  python 04_query.py     or     streamlit run app.py")
