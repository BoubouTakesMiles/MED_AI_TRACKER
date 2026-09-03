"""
STEP 2 - Load catalogued records into Qdrant.

Incremental: re-running after editing the spreadsheet embeds only new or
changed records, and removes records deleted from the spreadsheet.

  python 2_ingest_entities.py             # normal
  python 2_ingest_entities.py --rebuild   # force full re-embed

Use --rebuild only after changing build_text() or the embedding model, since
those invalidate every stored vector without changing the source text.

Writes dataset_quality_report.txt.
"""

import sys
from pathlib import Path

import pandas as pd
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pipeline_common import (connect, DATASET, SHEET, COLLECTION_ENTITIES, EMBED_MODEL,
                             VECTOR_SIZE, stable_id, content_hash,
                             existing_hashes, plan, report)

REPORT = Path("dataset_quality_report.txt")
REBUILD = "--rebuild" in sys.argv

KEYMAP = {
    "Country": "country", "Sector": "sector", "Sub-Sector": "sub_sector",
    "Entity Type": "entity_type", "Maturity": "maturity",
    "Entity / Innovation Name": "name", "Key Innovation / Description": "description",
    "Information Type": "info_type", "Reference Date": "reference_date",
    "Last Verified": "last_verified", "Recent Funding": "funding",
    "Source Name": "source_name", "Source Type": "source_type",
    "Source URL": "source_url", "Verification Status": "verification_status",
    "Notes": "notes",
}

ROLE = {
    "Startup": "{name} is a startup based in {country}, working in {sector} ({sub}).",
    "R&D": "{name} is a research institution, university laboratory or R&D initiative "
           "based in {country}, focused on {sector} ({sub}).",
    "Funding": "{name} is a venture capital firm or funding programme active in "
               "{country}, focused on {sector} ({sub}).",
    "Incubator/Accelerator": "{name} is a startup incubator or accelerator based in "
                             "{country}, supporting {sector} ({sub}).",
    "Policy/Strategy": "{name} is a government policy or strategic initiative in "
                       "{country}, related to {sector} ({sub}).",
    "Conference/Summit": "{name} is a conference or industry summit held in {country}, "
                         "focused on {sector} ({sub}).",
    "Hackathons": "{name} is a hackathon or AI competition held in {country}, focused "
                  "on {sector} ({sub}).",
    "Community/Association": "{name} is a professional community or association based "
                             "in {country}, focused on {sector} ({sub}).",
    "Other initiatives": "{name} is an initiative in {country}, related to {sector} ({sub}).",
}

BLANK = ("", "nan", "N/A", "Not specified", "None")


def build_text(row) -> str:
    """One readable sentence per record: questions are asked in prose, so
    passages shaped like prose retrieve better than field values."""
    tmpl = ROLE.get(row["Entity Type"],
                    "{name} is an entity based in {country}, related to {sector} ({sub}).")
    out = tmpl.format(name=row["Entity / Innovation Name"], country=row["Country"],
                      sector=row["Sector"], sub=row["Sub-Sector"])
    out += f" {str(row['Key Innovation / Description']).strip().rstrip('.')}."

    if str(row["Reference Date"]) not in BLANK:
        out += f" (reference date: {row['Reference Date']})"
    if str(row["Maturity"]) not in BLANK:
        out += f" This entity is at the '{row['Maturity']}' stage of maturity."

    fund = str(row["Recent Funding"])
    if fund == "Undisclosed":
        out += " Its funding amount is undisclosed."
    elif fund not in BLANK:
        out += f" It has raised {fund} in reported funding."

    out += f" Source: {row['Source Name']} ({row['Source URL']})."

    # The caveat is embedded in the text, not attached as metadata, so it
    # travels into the model's context automatically.
    if row["Verification Status"] == "Not yet verified":
        out += (" (Note: this entry has not yet been independently verified against "
                "a primary source.)")
    return out.strip()


def ensure_collection(client):
    if client.collection_exists(COLLECTION_ENTITIES):
        return
    client.create_collection(
        collection_name=COLLECTION_ENTITIES,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE))
    for field in ("country", "sector", "entity_type", "verification_status", "maturity"):
        client.create_payload_index(COLLECTION_ENTITIES, field_name=field,
                                    field_schema=PayloadSchemaType.KEYWORD)


def main():
    df = pd.read_excel(DATASET, sheet_name=SHEET).fillna("")
    print(f"Loaded {len(df)} records from {DATASET}.")
    print(f"Countries: {', '.join(sorted(df['Country'].unique()))}")

    flags = []
    for _, row in df.iterrows():
        nm = row["Entity / Innovation Name"]
        if not str(row["Source URL"]).startswith("http"):
            flags.append(f"missing or invalid source URL: {nm}")
        if str(row["Maturity"]) in BLANK:
            flags.append(f"maturity not recorded: {row['Country']} / {nm}")

    # records the cited source does not support are never indexed
    indexable = df[df["Verification Status"] != "Mismatch found"]
    excluded = len(df) - len(indexable)

    desired = {}
    for _, row in indexable.iterrows():
        pid = stable_id(row["Country"], row["Entity / Innovation Name"])
        if pid in desired:
            flags.append(f"DUPLICATE country+name, second ignored: "
                         f"{row['Country']} / {row['Entity / Innovation Name']}")
            continue
        text = build_text(row)
        desired[pid] = (content_hash(text), text, row)

    client = connect()
    if REBUILD and client.collection_exists(COLLECTION_ENTITIES):
        print("--rebuild: dropping the collection.")
        client.delete_collection(COLLECTION_ENTITIES)
    ensure_collection(client)

    stored = existing_hashes(client, COLLECTION_ENTITIES)
    to_write, to_delete, unchanged = plan({k: v[0] for k, v in desired.items()}, stored)
    print(f"{len(to_write)} to embed, {unchanged} unchanged, {len(to_delete)} to remove.")

    if to_write:
        print(f"Loading embedding model ({EMBED_MODEL})...")
        embed = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        points = []
        for n, pid in enumerate(to_write, 1):
            h, text, row = desired[pid]
            payload = {new: str(row[old]) for old, new in KEYMAP.items()
                       if old in df.columns}
            payload["_embedded_text"] = text
            payload["_hash"] = h
            points.append(PointStruct(id=pid, vector=embed.get_text_embedding(text),
                                      payload=payload))
            if n % 25 == 0:
                print(f"  {n}/{len(to_write)}")
        for s in range(0, len(points), 100):
            client.upsert(collection_name=COLLECTION_ENTITIES, points=points[s:s+100])
    if to_delete:
        client.delete(collection_name=COLLECTION_ENTITIES, points_selector=to_delete)
    if not to_write and not to_delete:
        print("Collection already matches the spreadsheet.")

    report("entities", len(to_write), len(to_delete), unchanged)
    print(f"  excluded (mismatch found): {excluded}")

    lines = [f"DATASET QUALITY REPORT - {DATASET}", "=" * 52,
             f"Total records : {len(df)}", f"Indexed       : {len(desired)}",
             f"Excluded      : {excluded} (verification status 'Mismatch found')", ""]
    for col, label in (("Country", "By country"), ("Sector", "By sector"),
                       ("Entity Type", "By entity type"),
                       ("Verification Status", "By verification status"),
                       ("Source Type", "By source type")):
        lines.append(f"{label}:")
        lines += [f"  {k}: {v}" for k, v in sorted(df[col].value_counts().items())]
        lines.append("")
    lines.append(f"Flags ({len(flags)}):")
    lines += [f"  - {f}" for f in flags] if flags else ["  none"]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nQuality report: {REPORT}")


if __name__ == "__main__":
    main()
