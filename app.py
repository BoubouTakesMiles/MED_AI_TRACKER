"""
STREAMLIT FRONT-END for the AI tracker.

Run with:   streamlit run app.py
(NOT "python app.py" - streamlit needs its own runner.)

Tabs:
  1. Search   - dropdown filters + semantic search + Kimi answer with sources
  2. Map      - 2D projection of the 376 entity embeddings, coloured by sector
  3. Overview - counts by country / sector / verification status

Needs: Qdrant running, step 02 (and ideally 03) done, key in .env.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------------------------------------------------------- config
st.set_page_config(page_title="MENA AI Ecosystem Tracker", layout="wide")

COUNTRIES = ["All", "Algeria", "Egypt", "Lebanon", "Morocco", "Tunisia"]
SECTORS = ["All", "Agriculture", "Cross-sector", "Education", "Energy",
           "Finance", "Government", "Healthcare", "Industry", "Maritime"]
ENTITY_TYPES = ["All", "Startup", "Hub", "Investor", "Policy"]

load_dotenv()
API_KEY = os.getenv("MOONSHOT_API_KEY")


# ------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading embedding model (first run is slow)...")
def get_embed():
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")


@st.cache_resource
def get_qdrant():
    return qdrant_client.QdrantClient(host="localhost", port=6333)


@st.cache_resource
def get_kimi():
    return OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1") if API_KEY else None


@st.cache_data(show_spinner="Pulling entities from Qdrant...")
def load_entities():
    """Every point + its vector, as a dataframe."""
    client = get_qdrant()
    records, offset = [], None
    while True:
        batch, offset = client.scroll("ai_entities", limit=256, offset=offset,
                                      with_payload=True, with_vectors=True)
        records.extend(batch)
        if offset is None:
            break
    rows = [{**r.payload, "_vector": r.vector} for r in records]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Projecting 1024-dim embeddings into 2D...")
def project_2d(vectors: np.ndarray, method: str):
    """UMAP if available, otherwise t-SNE, otherwise PCA."""
    if method == "UMAP":
        try:
            import umap
            return umap.UMAP(n_neighbors=15, min_dist=0.1,
                             metric="cosine", random_state=42).fit_transform(vectors)
        except ImportError:
            st.info("umap-learn not installed, falling back to t-SNE.")
            method = "t-SNE"
    if method == "t-SNE":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, metric="cosine",
                    init="pca", random_state=42).fit_transform(vectors)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(vectors)


def build_filter(country, sector, etype, field_map):
    conds = []
    for value, field in ((country, field_map["country"]),
                         (sector, field_map["sector"]),
                         (etype, field_map.get("entity_type"))):
        if value and value != "All" and field:
            conds.append(FieldCondition(key=field, match=MatchValue(value=value)))
    return Filter(must=conds) if conds else None


# ------------------------------------------------------------------ head
st.title("MENA AI Ecosystem Tracker")
st.caption("Morocco · Algeria · Tunisia · Lebanon · Egypt — "
           "376 curated entities with sourced, filterable retrieval")

try:
    df = load_entities()
except Exception as e:
    st.error(f"Could not reach Qdrant collection 'ai_entities'. "
             f"Is Docker running and step 02 done?\n\n{e}")
    st.stop()

tab_search, tab_map, tab_overview = st.tabs(["🔎 Search", "🗺️ Embedding map", "📊 Overview"])

# ---------------------------------------------------------------- SEARCH
with tab_search:
    c1, c2, c3 = st.columns(3)
    country = c1.selectbox("Country", COUNTRIES)
    sector = c2.selectbox("Application sector", SECTORS)
    etype = c3.selectbox("Entity type", ENTITY_TYPES)

    question = st.text_input("Question",
                             placeholder="Which startups work on medical imaging?")
    use_pages = st.checkbox("Also search full source-page content", value=True,
                            help="Requires step 03 (page ingestion) to have been run.")
    go = st.button("Search", type="primary")

    if go and question.strip():
        embed, qdrant = get_embed(), get_qdrant()
        qvec = embed.get_text_embedding(question)

        ents = qdrant.query_points(
            "ai_entities", query=qvec, limit=8, with_payload=True,
            query_filter=build_filter(country, sector, etype,
                                      {"country": "country",
                                       "sector": "application_sector",
                                       "entity_type": "entity_type"}),
        ).points

        chunks = []
        if use_pages and qdrant.collection_exists("ai_pages"):
            chunks = qdrant.query_points(
                "ai_pages", query=qvec, limit=4, with_payload=True,
                query_filter=build_filter(country, sector, None,
                                          {"country": "countries",
                                           "sector": "app_sectors"}),
            ).points

        if not ents:
            st.warning("No entities match those filters. Try widening them.")
        else:
            ctx = ["ENTITIES (structured tracker rows):"]
            for h in ents:
                p = h.payload
                ctx.append(f"- {p['name']} ({p['country']}, {p['application_sector']}, "
                           f"{p['entity_type']}): {p['description']} "
                           f"[Funding: {p.get('funding') or 'N/A'}; "
                           f"Status: {p['verification_status']}; Source: {p['source_url']}]")
            if chunks:
                ctx.append("\nSOURCE PAGE EXCERPTS:")
                for h in chunks:
                    p = h.payload
                    ctx.append(f"- (from {p['url']}): {p['text'][:700]}")
            context = "\n".join(ctx)

            kimi = get_kimi()
            if kimi is None:
                st.warning("No MOONSHOT_API_KEY in .env — showing retrieved entities only.")
            else:
                prompt = (
                    "You are a research assistant for an AI-ecosystem tracker covering "
                    "Morocco, Algeria, Tunisia, Lebanon and Egypt. Answer using ONLY the "
                    "material below. Do not speculate, do not add entities not listed, and "
                    "do not embellish descriptions. If the answer isn't in the material, "
                    "say so plainly. Cite entity names, countries and source URLs.\n\n"
                    f"{context}\n\nQUESTION: {question}"
                )

                def stream():
                    resp = kimi.chat.completions.create(
                        model="kimi-k2.6",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1,   # this model only accepts 1
                        stream=True,
                    )
                    for chunk in resp:
                        d = chunk.choices[0].delta.content
                        if d:
                            yield d

                st.subheader("Answer")
                try:
                    st.write_stream(stream)
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

            st.subheader("Entities retrieved")
            for h in ents:
                p = h.payload
                with st.expander(f"{p['name']} — {p['country']} · "
                                 f"{p['application_sector']} · score {h.score:.3f}"):
                    st.write(p["description"])
                    meta = st.columns(3)
                    meta[0].markdown(f"**Type**  \n{p['entity_type']}")
                    meta[1].markdown(f"**Funding**  \n{p.get('funding') or 'N/A'}")
                    meta[2].markdown(f"**Status**  \n{p['verification_status']}")
                    if p.get("source_url"):
                        st.markdown(f"[Source]({p['source_url']})")

            if chunks:
                st.subheader("Source-page excerpts")
                for h in chunks:
                    p = h.payload
                    with st.expander(f"{p['url'][:80]} — score {h.score:.3f}"):
                        st.write(p["text"][:1200])

# ------------------------------------------------------------------- MAP
with tab_map:
    st.markdown("Each point is one entity, positioned by the meaning of its "
                "description. Clusters are entities doing similar things — "
                "they were never told which sector they belong to.")

    m1, m2 = st.columns([1, 2])
    method = m1.selectbox("Projection", ["UMAP", "t-SNE", "PCA"])
    color_by = m2.selectbox("Colour by",
                            ["application_sector", "country", "entity_type",
                             "verification_status"])

    vectors = np.vstack(df["_vector"].values)
    coords = project_2d(vectors, method)

    plot_df = df.drop(columns=["_vector"]).copy()
    plot_df["x"], plot_df["y"] = coords[:, 0], coords[:, 1]
    plot_df["short"] = plot_df["description"].str.slice(0, 140) + "..."

    fig = px.scatter(
        plot_df, x="x", y="y", color=color_by,
        hover_name="name",
        hover_data={"country": True, "application_sector": True,
                    "entity_type": True, "short": True,
                    "x": False, "y": False},
        height=680,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8,
                                  line=dict(width=0.5, color="white")))
    fig.update_layout(xaxis_title=None, yaxis_title=None,
                      legend_title=color_by.replace("_", " ").title())
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Axes carry no units — only relative distance is meaningful. "
               "UMAP preserves neighbourhoods, t-SNE local structure, PCA "
               "global variance.")

# -------------------------------------------------------------- OVERVIEW
with tab_overview:
    a, b, c, d = st.columns(4)
    a.metric("Entities", len(df))
    b.metric("Countries", df["country"].nunique())
    c.metric("Verified", int((df["verification_status"] == "Verified").sum()))
    d.metric("Startups", int((df["entity_type"] == "Startup").sum()))

    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(df["country"].value_counts().sort_values(),
               orientation="h", title="Entities by country",
               labels={"value": "count", "index": ""}),
        use_container_width=True)
    right.plotly_chart(
        px.bar(df["application_sector"].value_counts().sort_values(),
               orientation="h", title="Entities by application sector",
               labels={"value": "count", "index": ""}),
        use_container_width=True)

    st.subheader("Coverage matrix — country × sector")
    st.caption("Empty and thin cells are where the tracker still has gaps.")
    matrix = pd.crosstab(df["country"], df["application_sector"])
    st.plotly_chart(
        px.imshow(matrix, text_auto=True, aspect="auto",
                  color_continuous_scale="Blues", height=380),
        use_container_width=True)

    st.subheader("Browse the full dataset")
    st.dataframe(
        df.drop(columns=["_vector", "_embedded_text"], errors="ignore"),
        use_container_width=True, height=420)
