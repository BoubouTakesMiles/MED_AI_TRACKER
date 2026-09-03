"""
Run the whole pipeline. This is the "I edited the spreadsheet, update
everything" command.

  python run_pipeline.py               # incremental
  python run_pipeline.py --rebuild     # force full re-embed
  python run_pipeline.py --skip-fetch  # no network access

Every step is safe to re-run: fetching skips cached URLs, ingestion skips
unchanged records.
"""

import subprocess
import sys
import time

REBUILD = ["--rebuild"] if "--rebuild" in sys.argv else []
steps = []
if "--skip-fetch" not in sys.argv:
    steps.append(("Fetch source pages", ["1_fetch_pages.py"]))
steps += [("Ingest records", ["2_ingest_entities.py"] + REBUILD),
          ("Ingest page chunks", ["3_ingest_pages.py"] + REBUILD)]

t0 = time.time()
for i, (label, cmd) in enumerate(steps, 1):
    print(f"\n{'=' * 66}\n[{i}/{len(steps)}] {label}\n{'=' * 66}")
    started = time.time()
    if subprocess.run([sys.executable] + cmd).returncode != 0:
        print(f"\nStopped: '{label}' failed. Fix the error above and re-run; "
              f"completed steps will not redo their work.")
        sys.exit(1)
    print(f"[{label}: {time.time() - started:.0f}s]")

print(f"\nPipeline complete in {time.time() - t0:.0f}s.")
print("Query with:  python query.py     or     streamlit run app.py")
