"""
STEP 4 - Ask questions. Searches both collections, Kimi answers.

v2 changes (adopted from the parallel pipeline):
  - DIVERSITY-CAPPED RETRIEVAL: over-fetch 30, then allow at most 2 chunks per
    source URL before keeping the top 10. Without this, one long article can
    occupy every slot and crowd out other entities. This was a real bug in v1.
  - Much stricter grounding prompt: flag source disagreements, surface
    verification status, hedge on "latest", answer in the question's language.
  - Answer shapes: LIST / DETAIL / COMPARATIVE.

Filters: country=Lebanon sector=Healthcare <question>
"""

import os
import re

import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise SystemExit("No MOONSHOT_API_KEY found. Copy .env.example to .env and add your key.")

COUNTRIES = ["Algeria", "Egypt", "France", "Italy", "Lebanon", "Morocco", "Spain", "Tunisia"]
SECTORS = ["Agriculture", "Cross-sector", "Education", "Energy", "Finance",
           "Government", "Healthcare", "Industry", "Maritime"]

OVER_FETCH = 30
MAX_PER_SOURCE = 2
TOP_ENTITIES = 10
TOP_CHUNKS = 4

SYSTEM_PROMPT = """You are a research assistant answering questions about the AI \
ecosystem, AI startups and AI innovation across the Mediterranean: Morocco, Algeria, \
Tunisia, Lebanon, Egypt, France, Spain and Italy.

GROUNDING RULES:
1. Answer ONLY from the provided context. Do not use outside knowledge, even if you \
believe you know the answer.
2. If the context only partly answers the question, give what you have and state \
plainly what is missing, rather than guessing or declining entirely.
3. Every claim must be attributable to a source in the context. Cite the source URL \
after each claim.
4. If an entry is marked "Not yet verified", say so rather than presenting it with \
full confidence.
5. If two sources in the current context disagree about the same entity (different \
funding figures, dates, partners), do NOT silently pick one. Name the discrepancy and \
attribute each version to its source. Compare actively; do not rely on a pre-existing \
flag to tell you there is a conflict.
6. Be concise and factual. No editorialising beyond what the sources support.
7. If asked about something outside this dataset's scope, say so rather than answering \
from general knowledge.
8. This dataset is a periodic snapshot, not live data. For "latest" or "most recent" \
questions, cite the most recent dated entry you actually have rather than asserting it \
is definitively the newest thing that exists.
9. If an answer is uneven across countries or entities (some have data, some don't), \
say so plainly instead of glossing over the gap.
10. Always respond in the same language the question was asked in.

ANSWER SHAPE - pick the one that fits:
MODE A - LIST (broad category question): up to 10 relevant entities, most relevant \
first, each with name, short description, country and sector. Never pad toward 10 with \
tangential results. If an entity appears in several chunks, list it once, merged.
MODE B - DETAIL (one specific entity): open with Entity Type, Sector, Country, \
Verification Status, then go deep on funding, founders, partnerships, dates.
MODE C - COMPARATIVE (comparing countries or entities): same structured treatment for \
each side, then a short synthesis of the actual similarities and differences. If \
coverage is uneven, say so as part of the synthesis."""

print("Loading embedding model...")
embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)
kimi = OpenAI(api_key=api_key, base_url="https://api.moonshot.ai/v1")
HAS_PAGES = qdrant.collection_exists("ai_pages")


def parse_filters(raw: str):
    filters, q = {}, raw
    for key, f_ent, f_page in (("country", "country", "countries"),
                               ("sector", "application_sector", "app_sectors")):
        m = re.search(rf"\b{key}=(\S+)", q, re.IGNORECASE)
        if m:
            val = m.group(1).replace("_", " ").title()
            pool = COUNTRIES if key == "country" else SECTORS
            match = next((v for v in pool if v.lower() == val.lower()), None)
            if match is None:
                print(f"WARNING: '{m.group(1)}' is not a known {key} "
                      f"({', '.join(pool)}). Ignoring this filter rather than "
                      f"silently returning zero results.")
            else:
                filters[key] = (match, f_ent, f_page)
            q = q.replace(m.group(0), "").strip()
    return filters, q


def qfilter(filters, which):
    conds = [FieldCondition(key=v[which], match=MatchValue(value=v[0]))
             for v in filters.values()]
    return Filter(must=conds) if conds else None


def retrieve_diverse(collection, qvec, qf, top_k, url_key):
    """Over-fetch, then cap how many hits any single source can contribute."""
    raw = qdrant.query_points(collection, query=qvec, limit=OVER_FETCH,
                              with_payload=True, query_filter=qf).points
    counts, kept = {}, []
    for h in raw:
        src = h.payload.get(url_key, "unknown")
        if counts.get(src, 0) >= MAX_PER_SOURCE:
            continue
        counts[src] = counts.get(src, 0) + 1
        kept.append(h)
        if len(kept) >= top_k:
            break
    return kept


print("Ready. Optional prefix: country=France sector=Healthcare  (or 'quit')\n")
while True:
    raw = input("Question: ").strip()
    if raw.lower() in ("quit", "exit", ""):
        break
    filters, q = parse_filters(raw)
    if filters:
        print("  filters:", {k: v[0] for k, v in filters.items()})

    qvec = embed.get_text_embedding(q)
    ents = retrieve_diverse("ai_entities", qvec, qfilter(filters, 1),
                            TOP_ENTITIES, "source_url")
    chunks = retrieve_diverse("ai_pages", qvec, qfilter(filters, 2),
                              TOP_CHUNKS, "url") if HAS_PAGES else []

    if not ents and not chunks:
        print("No relevant information found for those filters.\n")
        continue

    ctx = ["ENTITIES (structured tracker rows):"]
    for i, h in enumerate(ents, 1):
        p = h.payload
        ctx.append(f"[Source {i}, status: {p['verification_status']}]\n{p['_embedded_text']}")
    if chunks:
        ctx.append("\nSOURCE PAGE EXCERPTS:")
        for j, h in enumerate(chunks, len(ents) + 1):
            p = h.payload
            ctx.append(f"[Source {j}, full text from {p['url']}]\n{p['text'][:900]}")

    resp = kimi.chat.completions.create(
        model="kimi-k2.6",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": "Context:\n" + "\n\n".join(ctx) +
                                              f"\n\nQuestion: {q}"}],
        temperature=1,  # kimi-k2.6 only accepts 1
        stream=True,
    )
    print("\n=== ANSWER ===")
    for chunk in resp:
        d = chunk.choices[0].delta.content
        if d:
            print(d, end="", flush=True)

    print("\n\n=== RETRIEVED ===")
    for h in ents:
        print(f"[entity {h.score:.3f}] {h.payload['name']} ({h.payload['country']}) "
              f"- {h.payload['verification_status']}")
    for h in chunks:
        print(f"[page   {h.score:.3f}] {h.payload['url'][:80]}")
    print()
