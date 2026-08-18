"""
STEP 4 - Ask questions. Searches BOTH collections and has Kimi answer.

  - ai_entities : the structured rows (who exists, funding, status)
  - ai_pages    : full source-page chunks (the depth)

Optional filters, typed before your question:
  country=Lebanon sector=Healthcare What diagnostics startups exist?
  (sector = Application Sector: Education, Healthcare, Agriculture,
   Energy, Industry, Maritime, Finance, Government, Cross-sector)

Needs: Qdrant running, steps 2 (and ideally 3) done, key in .env.
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

print("Loading embedding model...")
embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)
kimi = OpenAI(api_key=api_key, base_url="https://api.moonshot.ai/v1")

TOP_ENTITIES = 6
TOP_CHUNKS = 4
HAS_PAGES = qdrant.collection_exists("ai_pages")


def parse_filters(raw: str):
    """Pull country=X / sector=Y off the front of the question."""
    filters, question = {}, raw
    for key, field_e, field_p in (("country", "country", "countries"),
                                  ("sector", "application_sector", "app_sectors")):
        m = re.search(rf"{key}=(\S+)", question, re.IGNORECASE)
        if m:
            filters[key] = (m.group(1).replace("_", " ").title(), field_e, field_p)
            question = question.replace(m.group(0), "").strip()
    return filters, question


def qfilter(filters, which):  # which: 1 = entity field, 2 = page field
    conds = [FieldCondition(key=v[which], match=MatchValue(value=v[0]))
             for v in filters.values()]
    return Filter(must=conds) if conds else None


print("Ready. Optional prefix: country=Lebanon sector=Finance  (or 'quit')\n")
while True:
    raw = input("Question: ").strip()
    if raw.lower() in ("quit", "exit", ""):
        break
    filters, q = parse_filters(raw)
    if filters:
        print("  filters:", {k: v[0] for k, v in filters.items()})
    qvec = embed.get_text_embedding(q)

    # 1) structured entities
    ents = qdrant.query_points("ai_entities", query=qvec, limit=TOP_ENTITIES,
                               with_payload=True, query_filter=qfilter(filters, 1)).points
    # 2) page chunks (if step 3 was run)
    chunks = []
    if HAS_PAGES:
        chunks = qdrant.query_points("ai_pages", query=qvec, limit=TOP_CHUNKS,
                                     with_payload=True, query_filter=qfilter(filters, 2)).points

    ctx = ["ENTITIES (structured tracker rows):"]
    for h in ents:
        p = h.payload
        ctx.append(f"- {p['name']} ({p['country']}, {p['application_sector']}, {p['entity_type']}): "
                   f"{p['description']} [Funding: {p['funding'] or 'N/A'}; "
                   f"Status: {p['verification_status']}; Source: {p['source_url']}]")
    if chunks:
        ctx.append("\nSOURCE PAGE EXCERPTS:")
        for h in chunks:
            p = h.payload
            ctx.append(f"- (from {p['url']}, re: {', '.join(p['entities'][:3])}): "
                       f"{p['text'][:700]}")
    context = "\n".join(ctx)

    prompt = (
        "You are a research assistant for an AI-ecosystem tracker covering "
        "Morocco, Algeria, Tunisia, Lebanon and Egypt. Answer using ONLY the "
        "material below. Do not speculate, do not add entities not listed, and "
        "do not embellish descriptions. If the answer isn't in the material, "
        "say so plainly. Cite entity names, countries and source URLs.\n\n"
        f"{context}\n\nQUESTION: {q}"
    )

    resp = kimi.chat.completions.create(
        model="kimi-k2.6",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,  # factual Q&A: keep it low, not 1
        stream=True,
    )
    print("\n=== ANSWER ===")
    for chunk in resp:
        d = chunk.choices[0].delta.content
        if d:
            print(d, end="", flush=True)
    print("\n\n=== RETRIEVED ===")
    for h in ents:
        print(f"[entity {h.score:.3f}] {h.payload['name']} ({h.payload['country']})")
    for h in chunks:
        print(f"[page   {h.score:.3f}] {h.payload['url'][:80]}")
    print()
