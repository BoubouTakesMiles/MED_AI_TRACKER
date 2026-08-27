"""
Shared helpers for incremental ingestion.

The whole idea: a point's ID is derived from WHAT IT IS (country + name),
never from where it sits in the spreadsheet. Row 3 and row 400 of the same
entity produce the same ID, so re-running never duplicates and never shifts.

A content hash stored in the payload lets the next run answer "did this
change?" without keeping any state outside Qdrant itself.
"""

import hashlib
import uuid

# Any fixed UUID works; it just has to never change once you've ingested.
NAMESPACE = uuid.UUID("6f1a3d20-0b7c-4f8e-9a2b-3c4d5e6f7a8b")


def stable_id(*parts) -> str:
    """Deterministic point ID from identifying fields.

    stable_id("Lebanon", "Berytech") always returns the same UUID.
    """
    key = "||".join(str(p).strip().lower() for p in parts)
    return str(uuid.uuid5(NAMESPACE, key))


def content_hash(text: str) -> str:
    """Short hash of the embedded text; changes when the content changes."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def existing_hashes(client, collection: str) -> dict:
    """Map {point_id: stored_hash} for everything currently in a collection.

    Returns {} if the collection doesn't exist yet.
    """
    if not client.collection_exists(collection):
        return {}
    out, offset = {}, None
    while True:
        batch, offset = client.scroll(
            collection, limit=256, offset=offset,
            with_payload=["_hash"], with_vectors=False,
        )
        for r in batch:
            out[str(r.id)] = (r.payload or {}).get("_hash")
        if offset is None:
            return out


def plan(current: dict, existing: dict):
    """Compare desired state against what's stored.

    current  = {point_id: hash} we want to exist
    existing = {point_id: hash} that already exists

    Returns (to_write, to_delete, unchanged_count).
    """
    to_write = [pid for pid, h in current.items() if existing.get(pid) != h]
    to_delete = [pid for pid in existing if pid not in current]
    unchanged = len(current) - len(to_write)
    return to_write, to_delete, unchanged


def report(label, written, deleted, unchanged):
    print(f"\n--- {label} ---")
    print(f"  added/updated : {written}")
    print(f"  unchanged     : {unchanged}  (skipped, not re-embedded)")
    print(f"  removed       : {deleted}")
