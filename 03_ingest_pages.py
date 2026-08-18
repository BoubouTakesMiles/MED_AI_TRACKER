"""
STEP 3 - Load the FETCHED PAGE CONTENT into Qdrant (second collection).

This is the semantic-depth layer: full articles / strategy PDFs, chunked.
The crucial move: every chunk inherits the metadata of the dataset row(s)
that cited its URL (country, application sector, entity name...), so a
semantic hit on page text can still be filtered by "Lebanon + Healthcare".

Run AFTER 01_fetch_pages.py and 02_ingest_entities.py.
Re-runnable: drops and rebuilds the collection.
"""

import hashlib
from pathlib import Path

import pandas as pd
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

XLSX = "tracking_dataset_v4.xlsx"
CORPUS = Path("corpus")
COLLECTION = "ai_pages"


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


df = pd.read_excel(XLSX, sheet_name="Dataset").fillna("")

# map each URL -> the metadata of the rows that cite it
url_meta = {}
for _, row in df.iterrows():
    url = str(row["Source URL"]).strip()
    if not url.startswith("http"):
        continue
    m = url_meta.setdefault(url, {"countries": set(), "app_sectors": set(), "entities": set()})
    m["countries"].add(str(row["Country"]))
    m["app_sectors"].add(str(row["Application Sector"]))
    m["entities"].add(str(row["Entity / Innovation Name"]))

print("Loading embedding model...")
embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)

client = qdrant_client.QdrantClient(host="localhost", port=6333)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
# country/app_sector are LISTS here (one page can back rows from
# several countries) - Qdrant keyword filters match "value in list"
for field in ("countries", "app_sectors"):
    client.create_payload_index(COLLECTION, field_name=field,
                                field_schema=PayloadSchemaType.KEYWORD)

points, pid = [], 0
files = 0
for url, meta in url_meta.items():
    txt_file = CORPUS / f"{url_id(url)}.txt"
    if not txt_file.exists():
        continue  # skipped / failed fetch - fine
    files += 1
    text = txt_file.read_text(encoding="utf-8")
    for chunk in splitter.split_text(text):
        payload = {
            "url": url,
            "countries": sorted(meta["countries"]),
            "app_sectors": sorted(meta["app_sectors"]),
            "entities": sorted(meta["entities"]),
            "text": chunk,
        }
        points.append(PointStruct(id=pid, vector=embed.get_text_embedding(chunk), payload=payload))
        pid += 1
    if files % 20 == 0:
        print(f"  {files} pages -> {pid} chunks so far...")

for start in range(0, len(points), 100):
    client.upsert(collection_name=COLLECTION, points=points[start:start + 100])

print(f"\nDone. {files} pages -> {pid} chunks in '{COLLECTION}'.")
