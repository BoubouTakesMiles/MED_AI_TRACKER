"""
Shared configuration and helpers.

DATASET is defined here and imported everywhere, so the filename lives in
exactly one place.

Incremental ingestion rests on two ideas: a point's ID is derived from what the
record IS (country + name), never from its row position, so re-running never
duplicates and inserting rows never scrambles anything; and a content hash
stored in the payload lets the next run answer "did this change?" without
keeping state outside Qdrant.
"""

import hashlib
import uuid

DATASET = "tracking_dataset.xlsx"
SHEET = "Dataset"
COLLECTION_ENTITIES = "ai_entities"
COLLECTION_PAGES = "ai_pages"
EMBED_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024

# Fixed once ingested; changing it orphans every existing point.
NAMESPACE = uuid.UUID("6f1a3d20-0b7c-4f8e-9a2b-3c4d5e6f7a8b")


def stable_id(*parts) -> str:
    """Deterministic point ID. stable_id("Lebanon", "Berytech") is always the same."""
    key = "||".join(str(p).strip().lower() for p in parts)
    return str(uuid.uuid5(NAMESPACE, key))


def content_hash(text: str) -> str:
    """Short hash of embedded text; changes when the content changes."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def existing_hashes(client, collection: str) -> dict:
    """{point_id: stored_hash} for a collection, or {} if it does not exist."""
    if not client.collection_exists(collection):
        return {}
    out, offset = {}, None
    while True:
        batch, offset = client.scroll(collection, limit=256, offset=offset,
                                      with_payload=["_hash"], with_vectors=False)
        for r in batch:
            out[str(r.id)] = (r.payload or {}).get("_hash")
        if offset is None:
            return out


def plan(current: dict, existing: dict):
    """Compare desired state to stored state -> (to_write, to_delete, unchanged)."""
    to_write = [pid for pid, h in current.items() if existing.get(pid) != h]
    to_delete = [pid for pid in existing if pid not in current]
    return to_write, to_delete, len(current) - len(to_write)


def report(label, written, deleted, unchanged):
    print(f"\n--- {label} ---")
    print(f"  added/updated : {written}")
    print(f"  unchanged     : {unchanged}  (skipped, not re-embedded)")
    print(f"  removed       : {deleted}")


def connect(host: str = "localhost", port: int = 6333):
    """Qdrant client, with a readable message instead of a 40-line traceback."""
    import sys
    import qdrant_client
    client = qdrant_client.QdrantClient(host=host, port=port)
    try:
        client.get_collections()
    except Exception:
        print(f"\nCannot reach Qdrant at {host}:{port}.\n"
              "  1. Is Docker Desktop running?\n"
              "  2. Is the container started?   docker ps -a   then   "
              "docker start <id>\n"
              "  3. Check http://localhost:6333/dashboard in a browser.\n"
              "\nTo make the container restart automatically:\n"
              "  docker update --restart unless-stopped <id>\n", file=sys.stderr)
        sys.exit(1)
    return client
