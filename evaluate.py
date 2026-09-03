"""
VALIDATION - measures whether the system actually works.

Writes EVERY question to eval_results.json, both the automatically scored ones
and the ones needing human judgement. Manual questions are written with their
retrieved results and verdict=null; you grade them in the app's Validation tab,
which saves verdicts to eval_verdicts.json and folds them into the headline
numbers.

  python evaluate.py

Outputs:
  eval_results.json   consumed by app.py
  eval_report.txt     plain text summary
"""

import json
import statistics
from pathlib import Path

import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline_common import (connect, COLLECTION_ENTITIES as COLLECTION,
                             EMBED_MODEL)

EVAL_PATH = Path("eval_questions.json")
TOP_K = 5
OVER_FETCH = 30
MAX_PER_SOURCE = 2


def retrieve(client, embed, question, country=None, k=TOP_K):
    qf = None
    if country:
        qf = Filter(must=[FieldCondition(key="country",
                                         match=MatchValue(value=country))])
    qvec = embed.get_text_embedding(question)
    raw = client.query_points(COLLECTION, query=qvec, limit=OVER_FETCH,
                              with_payload=True, query_filter=qf).points
    counts, kept = {}, []
    for h in raw:
        src = h.payload.get("source_url", "unknown")
        if counts.get(src, 0) >= MAX_PER_SOURCE:
            continue
        counts[src] = counts.get(src, 0) + 1
        kept.append(h)
        if len(kept) >= k:
            break
    return kept


def main():
    spec = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = spec["questions"]

    print("Loading embedding model...")
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    client = connect()
    total = client.count(COLLECTION).count
    print(f"Collection '{COLLECTION}': {total} entities\n")

    results, recalls, in_scope_top, oos_top = [], [], [], []

    for q in questions:
        hits = retrieve(client, embed, q["question"], q["country_filter"])
        names = [h.payload.get("name", "?") for h in hits]
        scores = [round(h.score, 3) for h in hits]
        countries = [h.payload.get("country", "?") for h in hits]
        snippets = [h.payload.get("description", "")[:180] for h in hits]

        expected = q["expected_names"]
        auto = bool(expected)

        entry = {
            **q,
            "scoring": "automatic" if auto else "manual",
            "retrieved": names,
            "retrieved_countries": countries,
            "retrieved_snippets": snippets,
            "scores": scores,
            "top_score": scores[0] if scores else None,
        }

        if auto:
            found = [e for e in expected if e in names]
            recall = len(found) / len(expected)
            recalls.append(recall)
            if scores:
                in_scope_top.append(scores[0])
            entry["recall"] = recall
            entry["ranks"] = [names.index(e) + 1 for e in found]
            entry["missing"] = [e for e in expected if e not in found]
            verdict = "PASS" if recall == 1.0 else ("PARTIAL" if recall > 0 else "FAIL")
            print(f"[{q['id']}] {verdict:8} recall {recall:.2f}  {q['question'][:52]}")
        else:
            entry["recall"] = None
            entry["verdict"] = None          # filled in via the app
            entry["verdict_note"] = ""
            if q["category"] == "out_of_scope" and scores:
                oos_top.append(scores[0])
            print(f"[{q['id']}] MANUAL   top {scores[0] if scores else 0:.3f}  "
                  f"{q['question'][:52]}")

        results.append(entry)

    avg_recall = statistics.mean(recalls) if recalls else 0
    full = sum(1 for r in recalls if r == 1.0)
    ins = statistics.mean(in_scope_top) if in_scope_top else None
    oos = statistics.mean(oos_top) if oos_top else None

    print(f"\n{'-' * 70}")
    print(f"Automatic: average Recall@{TOP_K} = {avg_recall:.1%} "
          f"({full}/{len(recalls)} fully correct)")
    if ins is not None and oos is not None:
        print(f"Score separation: in-scope {ins:.3f} vs out-of-scope {oos:.3f} "
              f"= {ins - oos:+.3f}")
    print(f"Manual review required for "
          f"{sum(1 for r in results if r['scoring'] == 'manual')} questions.")
    print("Grade them in the app: streamlit run app.py -> Validation tab")

    out = {
        "collection_size": total,
        "top_k": TOP_K,
        "average_recall": avg_recall,
        "fully_correct": full,
        "scored_questions": len(recalls),
        "manual_questions": sum(1 for r in results if r["scoring"] == "manual"),
        "mean_top_score_in_scope": ins,
        "mean_top_score_out_of_scope": oos,
        "results": results,
    }
    Path("eval_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

    lines = ["RETRIEVAL VALIDATION REPORT", "=" * 46,
             f"Collection size        : {total} entities",
             f"Automatically scored   : {len(recalls)}",
             f"Average Recall@{TOP_K}      : {avg_recall:.1%}",
             f"Fully correct          : {full}/{len(recalls)}",
             f"Manual review required : {out['manual_questions']}", ""]
    if ins is not None and oos is not None:
        lines += [f"Mean top score in-scope     : {ins:.3f}",
                  f"Mean top score out-of-scope : {oos:.3f}",
                  f"Separation                  : {ins - oos:+.3f}", ""]
    lines.append("Per-question:")
    for r in results:
        rec = f"{r['recall']:.2f}" if r["recall"] is not None else "manual"
        lines.append(f"  {r['id']}  {rec:>6}  {r['question'][:54]}")
    Path("eval_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nWritten: eval_results.json, eval_report.txt")


if __name__ == "__main__":
    main()
