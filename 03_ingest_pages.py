"""
STEP 3 - Load fetched page content into Qdrant. INCREMENTAL.

Only chunks from newly fetched or changed pages get embedded. This is the
step that used to re-embed everything (20+ minutes); now adding 20 URLs
costs about a minute.

  python 03_ingest_pages.py             # incremental
  python 03_ingest_pages.py --rebuild   # wipe and redo everything

Run AFTER 01_fetch_pages.py.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline_common import stable_id, content_hash, existing_hashes, plan, report

XLSX = "tracking_dataset_v5.xlsx"
CORPUS = Path("corpus")
COLLECTION = "ai_pages"
REBUILD = "--rebuild" in sys.argv


def url_id(url: str) -> str:
    """Matches the filename scheme used by 01_fetch_pages.py."""
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def ensure_collection(client):
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    for field in ("countries", "app_sectors"):
        client.create_payload_index(COLLECTION, field_name=field,
                                    field_schema=PayloadSchemaType.KEYWORD)


def main():
    df = pd.read_excel(XLSX, sheet_name="Dataset").fillna("")

    url_meta = {}
    for _, row in df.iterrows():
        url = str(row["Source URL"]).strip()
        if not url.startswith("http"):
            continue
        m = url_meta.setdefault(url, {"countries": set(), "app_sectors": set(),
                                      "entities": set()})
        m["countries"].add(str(row["Country"]))
        m["app_sectors"].add(str(row["Application Sector"]))
        m["entities"].add(str(row["Entity / Innovation Name"]))

    splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)

    # desired state: chunk_id -> (hash, payload)
    desired, pages = {}, 0
    for url, meta in url_meta.items():
        f = CORPUS / f"{url_id(url)}.txt"
        if not f.exists():
            continue
        pages += 1
        text = f.read_text(encoding="utf-8")
        for i, chunk in enumerate(splitter.split_text(text)):
            cid = stable_id(url, i)
            payload = {
                "url": url,
                "countries": sorted(meta["countries"]),
                "app_sectors": sorted(meta["app_sectors"]),
                "entities": sorted(meta["entities"]),
                "text": chunk,
                "chunk_index": i,
            }
            # hash covers the chunk AND its metadata, so a row edit that
            # changes a page's country/sector tags triggers a refresh too
            payload["_hash"] = content_hash(
                chunk + "|" + ",".join(payload["countries"]) +
                "|" + ",".join(payload["app_sectors"]))
            desired[cid] = (payload["_hash"], payload)

    print(f"{pages} fetched pages -> {len(desired)} chunks desired.")

    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    if REBUILD and client.collection_exists(COLLECTION):
        print("--rebuild: dropping the collection.")
        client.delete_collection(COLLECTION)
    ensure_collection(client)

    stored = existing_hashes(client, COLLECTION)
    to_write, to_delete, unchanged = plan({k: v[0] for k, v in desired.items()}, stored)

    print(f"{len(to_write)} chunks to embed, {unchanged} unchanged, "
          f"{len(to_delete)} to remove.")
    if not to_write and not to_delete:
        print("Nothing to do. Collection already matches the corpus.")
    else:
        if to_write:
            print("Loading embedding model...")
            embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
            points = []
            for n, cid in enumerate(to_write, 1):
                _, payload = desired[cid]
                points.append(PointStruct(
                    id=cid,
                    vector=embed.get_text_embedding(payload["text"]),
                    payload=payload))
                if n % 50 == 0:
                    print(f"  {n}/{len(to_write)}")
            for s in range(0, len(points), 100):
                client.upsert(collection_name=COLLECTION, points=points[s:s + 100])
        if to_delete:
            client.delete(collection_name=COLLECTION, points_selector=to_delete)

    report("page chunks", len(to_write), len(to_delete), unchanged)


if __name__ == "__main__":
    main()
