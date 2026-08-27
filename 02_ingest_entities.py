"""
STEP 2 - Load entities into Qdrant. INCREMENTAL.

Re-running after adding rows to the spreadsheet only embeds the new or
changed ones. Everything else is left alone. Rows deleted from the xlsx are
removed from the collection.

  python 02_ingest_entities.py             # incremental (normal use)
  python 02_ingest_entities.py --rebuild   # wipe and redo everything

Use --rebuild only when you change build_text() or the embedding model,
since that invalidates every stored vector.
"""

import sys
from pathlib import Path

import pandas as pd
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline_common import stable_id, content_hash, existing_hashes, plan, report

XLSX = "tracking_dataset_v5.xlsx"
COLLECTION = "ai_entities"
REPORT = Path("validation_report.txt")
REBUILD = "--rebuild" in sys.argv

KEYMAP = {
    "Country": "country", "Sector": "sector", "Sub-Sector": "sub_sector",
    "Application Sector": "application_sector", "Entity Type": "entity_type",
    "Entity / Innovation Name": "name", "Key Innovation / Description": "description",
    "Information Type": "info_type", "Reference Date": "reference_date",
    "Last Verified": "last_verified", "Recent Funding": "funding",
    "Source Name": "source_name", "Source Type": "source_type",
    "Source URL": "source_url", "Verification Status": "verification_status",
    "Notes": "notes",
}

ROLE = {
    "Startup": "{name} is a startup based in {country}, working in {sector} ({sub}), serving the {app} sector.",
    "Hub": "{name} is a hub, incubator, research centre or ecosystem organisation in {country}, active in {sector} ({sub}), serving the {app} sector.",
    "Investor": "{name} is an investor or funding programme active in {country}, focused on {sector} ({sub}) in the {app} sector.",
    "Policy": "{name} is a government policy, strategy or public initiative in {country}, covering {sector} ({sub}) in the {app} sector.",
}


def build_text(row) -> str:
    tmpl = ROLE.get(row["Entity Type"],
                    "{name} is an entity based in {country}, related to {sector} ({sub}).")
    out = tmpl.format(name=row["Entity / Innovation Name"], country=row["Country"],
                      sector=row["Sector"], sub=row["Sub-Sector"],
                      app=row["Application Sector"])
    out += f" {str(row['Key Innovation / Description']).strip().rstrip('.')}."

    ref = str(row["Reference Date"])
    if ref and ref not in ("Not specified", "N/A", "nan", ""):
        out += f" (reference date: {ref})"

    fund = str(row["Recent Funding"])
    if fund == "Undisclosed":
        out += " Its funding amount is undisclosed."
    elif fund and fund not in ("N/A", "nan", ""):
        out += f" It has raised {fund} in reported funding."

    out += f" Source: {row['Source Name']} ({row['Source URL']})."

    if row["Verification Status"] == "Not yet verified":
        out += (" (Note: this entry has not yet been independently verified "
                "against a primary source.)")
    elif row["Verification Status"] == "Mismatch found":
        out += " (Note: the cited source does NOT confirm this claim.)"
    return out.strip()


def ensure_collection(client):
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    for field in ("country", "application_sector", "entity_type",
                  "verification_status", "sector"):
        client.create_payload_index(COLLECTION, field_name=field,
                                    field_schema=PayloadSchemaType.KEYWORD)


def main():
    df = pd.read_excel(XLSX, sheet_name="Dataset").fillna("")
    print(f"Loaded {len(df)} rows from {XLSX}.")

    flags = []
    for _, row in df.iterrows():
        name = row["Entity / Innovation Name"]
        if not str(row["Source URL"]).startswith("http"):
            flags.append(f"Missing/invalid source URL: {name}")
        if row["Application Sector"] == "":
            flags.append(f"Missing Application Sector: {name}")

    indexable = df[df["Verification Status"] != "Mismatch found"]
    excluded = len(df) - len(indexable)

    # desired state: id -> (hash, text, row)
    desired = {}
    for _, row in indexable.iterrows():
        pid = stable_id(row["Country"], row["Entity / Innovation Name"])
        text = build_text(row)
        if pid in desired:
            flags.append(f"DUPLICATE country+name (second one ignored): "
                         f"{row['Country']} / {row['Entity / Innovation Name']}")
            continue
        desired[pid] = (content_hash(text), text, row)

    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    if REBUILD and client.collection_exists(COLLECTION):
        print("--rebuild: dropping the collection.")
        client.delete_collection(COLLECTION)
    ensure_collection(client)

    stored = existing_hashes(client, COLLECTION)
    to_write, to_delete, unchanged = plan({k: v[0] for k, v in desired.items()}, stored)

    print(f"{len(to_write)} to embed, {unchanged} unchanged, {len(to_delete)} to remove.")
    if not to_write and not to_delete:
        print("Nothing to do. Collection already matches the spreadsheet.")
    else:
        if to_write:
            print("Loading embedding model...")
            embed = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
            points = []
            for n, pid in enumerate(to_write, 1):
                h, text, row = desired[pid]
                payload = {new: str(row[old]) for old, new in KEYMAP.items()
                           if old in df.columns}
                payload["_embedded_text"] = text
                payload["_hash"] = h
                points.append(PointStruct(id=pid,
                                          vector=embed.get_text_embedding(text),
                                          payload=payload))
                if n % 25 == 0:
                    print(f"  {n}/{len(to_write)}")
            for s in range(0, len(points), 100):
                client.upsert(collection_name=COLLECTION, points=points[s:s + 100])
        if to_delete:
            client.delete(collection_name=COLLECTION, points_selector=to_delete)

    report("entities", len(to_write), len(to_delete), unchanged)
    print(f"  excluded (mismatch found): {excluded}")

    lines = [f"Total rows: {len(df)}", f"Indexed: {len(desired)}",
             f"Excluded (mismatch found): {excluded}", ""]
    for col, label in (("Country", "By country"),
                       ("Verification Status", "By verification status"),
                       ("Entity Type", "By entity type"),
                       ("Application Sector", "By application sector")):
        lines.append(f"{label}:")
        lines += [f"  {k}: {v}" for k, v in sorted(df[col].value_counts().items())]
        lines.append("")
    lines += ["Flags:"] + ([f"  - {f}" for f in flags] if flags else ["  none"])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
