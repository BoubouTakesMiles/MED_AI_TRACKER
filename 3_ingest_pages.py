"""
STEP 3 - Load fetched page content into Qdrant as a second collection.

Each chunk inherits the metadata of the records citing its URL, so a semantic
match inside a long article stays filterable by country and sector.

  python 3_ingest_pages.py             # incremental
  python 3_ingest_pages.py --rebuild   # force full re-embed

Run after 1_fetch_pages.py.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline_common import (connect, DATASET, SHEET, COLLECTION_PAGES, EMBED_MODEL,
                             VECTOR_SIZE, stable_id, content_hash,
                             existing_hashes, plan, report)

CORPUS = Path("corpus")
REBUILD = "--rebuild" in sys.argv


def url_id(url: str) -> str:
    """Must match the scheme used by 1_fetch_pages.py."""
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def ensure_collection(client):
    if client.collection_exists(COLLECTION_PAGES):
        return
    client.create_collection(
        collection_name=COLLECTION_PAGES,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))
    for field in ("countries", "sectors"):
        client.create_payload_index(COLLECTION_PAGES, field_name=field,
                                    field_schema=PayloadSchemaType.KEYWORD)


def main():
    df = pd.read_excel(DATASET, sheet_name=SHEET).fillna("")

    url_meta = {}
    for _, row in df.iterrows():
        url = str(row["Source URL"]).strip()
        if not url.startswith("http"):
            continue
        m = url_meta.setdefault(url, {"countries": set(), "sectors": set(),
                                      "entities": set()})
        m["countries"].add(str(row["Country"]))
        m["sectors"].add(str(row["Sector"]))
        m["entities"].add(str(row["Entity / Innovation Name"]))

    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    desired, pages = {}, 0
    for url, meta in url_meta.items():
        f = CORPUS / f"{url_id(url)}.txt"
        if not f.exists():
            continue
        pages += 1
        for i, chunk in enumerate(splitter.split_text(f.read_text(encoding="utf-8"))):
            payload = {
                "url": url,
                "countries": sorted(meta["countries"]),
                "sectors": sorted(meta["sectors"]),
                "entities": sorted(meta["entities"]),
                "text": chunk,
                "chunk_index": i,
            }
            # hash covers metadata too, so retagging a record refreshes its chunks
            payload["_hash"] = content_hash(
                chunk + "|" + ",".join(payload["countries"])
                + "|" + ",".join(payload["sectors"]))
            desired[stable_id(url, i)] = (payload["_hash"], payload)

    print(f"{pages} fetched pages -> {len(desired)} chunks desired.")

    client = connect()
    if REBUILD and client.collection_exists(COLLECTION_PAGES):
        print("--rebuild: dropping the collection.")
        client.delete_collection(COLLECTION_PAGES)
    ensure_collection(client)

    stored = existing_hashes(client, COLLECTION_PAGES)
    to_write, to_delete, unchanged = plan({k: v[0] for k, v in desired.items()}, stored)
    print(f"{len(to_write)} chunks to embed, {unchanged} unchanged, "
          f"{len(to_delete)} to remove.")

    if to_write:
        print(f"Loading embedding model ({EMBED_MODEL})...")
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        points = []
        for n, cid in enumerate(to_write, 1):
            _, payload = desired[cid]
            points.append(PointStruct(id=cid,
                                      vector=embed.get_text_embedding(payload["text"]),
                                      payload=payload))
            if n % 50 == 0:
                print(f"  {n}/{len(to_write)}")
        for s in range(0, len(points), 100):
            client.upsert(collection_name=COLLECTION_PAGES, points=points[s:s+100])
    if to_delete:
        client.delete(collection_name=COLLECTION_PAGES, points_selector=to_delete)
    if not to_write and not to_delete:
        print("Collection already matches the corpus.")

    report("page chunks", len(to_write), len(to_delete), unchanged)


if __name__ == "__main__":
    main()
