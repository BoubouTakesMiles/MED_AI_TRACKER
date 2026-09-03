"""
Command-line query tool.

  python query.py                                    interactive
  python query.py "which startups do medical imaging"
  python query.py --country Egypt "AI in agriculture"
  python query.py --no-llm "AI in agriculture"       retrieval only, no API key

--no-llm prints the matching records with similarity scores and stops. It is
the way to check whether a poor answer is a retrieval problem or a generation
problem, and it needs no API key.

Requires Qdrant running and 2_ingest_entities.py completed.
"""

import os
import sys
import textwrap

import qdrant_client
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline_common import (connect, COLLECTION_ENTITIES,
                             COLLECTION_PAGES, EMBED_MODEL)

TOP_ENTITIES = 8
TOP_CHUNKS = 4
OVER_FETCH = 30
MAX_PER_SOURCE = 2
WIDTH = 78
CHAT_MODEL = "kimi-k2.6"

SYSTEM_PROMPT = """You are a research assistant answering questions about the AI \
ecosystem of the Mediterranean: Morocco, Algeria, Tunisia, Lebanon, Egypt, France, \
Spain and Italy.

Answer ONLY from the provided context. Do not use outside knowledge, even if you \
believe you know the answer. If the context only partly answers the question, give \
what you have and state plainly what is missing rather than guessing. Cite the source \
URL after each claim. If an entry is marked "Not yet verified", say so rather than \
presenting it with full confidence. If two sources disagree about the same entity, \
name the discrepancy and attribute each version rather than silently picking one. \
This dataset is a periodic snapshot: for "latest" or "most recent" questions, cite the \
most recent dated entry you actually have rather than asserting it is the newest thing \
that exists. If coverage is uneven across countries, say so. Answer in the language of \
the question. Be concise and factual.

Choose the answer shape that fits: a LIST of up to ten relevant entities for broad \
category questions, never padded with tangential results; a DETAIL answer opening with \
type, sector, country and verification status for questions about one entity; or a \
COMPARATIVE answer giving each side the same treatment and closing with a short \
synthesis, noting explicitly if coverage is uneven between them."""


def resolve(value, pool, label):
    if not value:
        return None
    for known in pool:
        if value.strip().lower() == known.lower():
            return known
    print(f"WARNING: '{value}' is not a known {label}. Known values: "
          f"{', '.join(pool)}.\nProceeding with no {label} filter rather than "
          f"silently returning zero results.\n")
    return None


def retrieve(client, embed, question, country=None, sector=None):
    """Over-fetch, then cap hits per source so one source cannot fill the page."""
    conds = []
    if country:
        conds.append(FieldCondition(key="country", match=MatchValue(value=country)))
    if sector:
        conds.append(FieldCondition(key="sector", match=MatchValue(value=sector)))
    qf = Filter(must=conds) if conds else None

    qvec = embed.get_text_embedding(question)
    raw = client.query_points(COLLECTION_ENTITIES, query=qvec, limit=OVER_FETCH,
                              with_payload=True, query_filter=qf).points
    counts, kept = {}, []
    for h in raw:
        src = h.payload.get("source_url", "unknown")
        if counts.get(src, 0) >= MAX_PER_SOURCE:
            continue
        counts[src] = counts.get(src, 0) + 1
        kept.append(h)
        if len(kept) >= TOP_ENTITIES:
            break

    chunks = []
    if client.collection_exists(COLLECTION_PAGES):
        pf = None
        if country:
            pf = Filter(must=[FieldCondition(key="countries",
                                             match=MatchValue(value=country))])
        chunks = client.query_points(COLLECTION_PAGES, query=qvec, limit=TOP_CHUNKS,
                                     with_payload=True, query_filter=pf).points
    return kept, chunks


def print_records(hits):
    print(f"\nTop {len(hits)} matching records")
    print("-" * WIDTH)
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"\n[{i}]  similarity {h.score:.3f}   |   "
              f"status: {p.get('verification_status', 'unknown')}")
        print(textwrap.fill(p.get("_embedded_text", p.get("description", "")),
                            width=WIDTH, initial_indent="     ",
                            subsequent_indent="     "))
        print(f"     Source: {p.get('source_url', 'n/a')}")
    print()


def answer(llm, hits, chunks, question):
    ctx = ["RECORDS:"]
    for i, h in enumerate(hits, 1):
        ctx.append(f"[Source {i}, status: {h.payload['verification_status']}]\n"
                   f"{h.payload['_embedded_text']}")
    if chunks:
        ctx.append("\nSOURCE PAGE EXCERPTS:")
        for j, h in enumerate(chunks, len(hits) + 1):
            ctx.append(f"[Source {j}, full text from {h.payload['url']}]\n"
                       f"{h.payload['text'][:900]}")

    resp = llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",
                   "content": "Context:\n" + "\n\n".join(ctx) +
                              f"\n\nQuestion: {question}"}],
        temperature=1,   # this model accepts only 1
        stream=True)

    print("\n=== ANSWER ===")
    for chunk in resp:
        d = chunk.choices[0].delta.content
        if d:
            print(d, end="", flush=True)

    print("\n\n=== SOURCES USED ===")
    for h in hits:
        print(f"[{h.score:.3f}] {h.payload['name']} ({h.payload['country']}) - "
              f"{h.payload['verification_status']} - {h.payload['source_url']}")
    for h in chunks:
        print(f"[{h.score:.3f}] page: {h.payload['url'][:70]}")
    print()


def main():
    args = sys.argv[1:]
    no_llm = "--no-llm" in args
    args = [a for a in args if a != "--no-llm"]

    country = sector = None
    for flag, setter in (("--country", "country"), ("--sector", "sector")):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1] if i + 1 < len(args) else None
            args = args[:i] + args[i + 2:]
            if setter == "country":
                country = val
            else:
                sector = val

    print(f"Loading embedding model ({EMBED_MODEL})...")
    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    client = connect()
    total = client.count(COLLECTION_ENTITIES).count
    print(f"Connected to '{COLLECTION_ENTITIES}': {total} records.")

    # vocabularies come from the collection, never a hardcoded list
    sample, _ = client.scroll(COLLECTION_ENTITIES, limit=1000,
                              with_payload=["country", "sector"], with_vectors=False)
    countries = sorted({p.payload["country"] for p in sample})
    sectors = sorted({p.payload["sector"] for p in sample})
    country = resolve(country, countries, "country")
    sector = resolve(sector, sectors, "sector")

    llm = None
    if not no_llm:
        load_dotenv()
        key = os.getenv("MOONSHOT_API_KEY")
        if not key:
            print("No MOONSHOT_API_KEY in .env - running in retrieval-only mode.")
            no_llm = True
        else:
            llm = OpenAI(api_key=key, base_url="https://api.moonshot.ai/v1")

    def run(q):
        hits, chunks = retrieve(client, embed, q, country, sector)
        if not hits:
            print("No records match. Try widening the filters.\n")
            return
        if no_llm:
            print_records(hits)
        else:
            answer(llm, hits, chunks, q)

    if args:
        run(" ".join(args))
    else:
        print("Interactive mode. Type a question, or 'exit'.")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q:
                run(q)


if __name__ == "__main__":
    main()
