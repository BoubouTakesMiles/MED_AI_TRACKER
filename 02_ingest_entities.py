"""
STEP 2 - Load the 376 entities into Qdrant as filterable points.

One point per row. The vector = embedding of "Name. Description."
The payload = every column, so you can FILTER (country, application
sector, entity type, verification status) - that's where most of the
tool's matching value lives, no scraping involved.

Re-runnable: drops and rebuilds the collection each time (it's fast).
"""

import pandas as pd
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

XLSX = "tracking_dataset_v4.xlsx"
COLLECTION = "ai_entities"

# clean payload keys (no spaces/slashes) - easier to filter on
KEYMAP = {
    "Country": "country",
    "Sector": "sector",
    "Sub-Sector": "sub_sector",
    "Application Sector": "application_sector",
    "Entity Type": "entity_type",
    "Entity / Innovation Name": "name",
    "Key Innovation / Description": "description",
    "Information Type": "info_type",
    "Reference Date": "reference_date",
    "Last Verified": "last_verified",
    "Recent Funding": "funding",
    "Source Name": "source_name",
    "Source Type": "source_type",
    "Source URL": "source_url",
    "Verification Status": "verification_status",
    "Notes": "notes",
}

df = pd.read_excel(XLSX, sheet_name="Dataset").fillna("")
print(f"Loaded {len(df)} rows.")

print("Loading embedding model...")
embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

client = qdrant_client.QdrantClient(host="localhost", port=6333)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# payload indexes: make the common filters fast + explicit
for field in ("country", "application_sector", "entity_type", "verification_status", "sector"):
    client.create_payload_index(COLLECTION, field_name=field,
                                field_schema=PayloadSchemaType.KEYWORD)

print("Embedding entities...")
points = []
for i, row in df.iterrows():
    text = f"{row['Entity / Innovation Name']}. {row['Key Innovation / Description']}".strip()
    payload = {new: str(row[old]) for old, new in KEYMAP.items() if old in df.columns}
    payload["_embedded_text"] = text
    points.append(PointStruct(id=int(i), vector=embed.get_text_embedding(text), payload=payload))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(df)}")

# upload in batches (one giant upsert can time out)
for start in range(0, len(points), 100):
    client.upsert(collection_name=COLLECTION, points=points[start:start + 100])

print(f"\nDone. {len(points)} entities in '{COLLECTION}' with filter indexes.")
