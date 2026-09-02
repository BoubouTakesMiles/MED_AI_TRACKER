"""
Streamlit interface for the AI Ecosystem Tracker.

Run:  streamlit run app.py     (NOT python app.py)

Tabs
  Search       filtered semantic search + grounded answer with sources
  Entity       full record view, source link, nearest neighbours in vector space
  Compare      two countries side by side
  Map          2D projection of the entity embeddings
  Overview     dataset composition and coverage gaps
  Validation   retrieval evaluation results

Expects the v3 collection built by ingest_v3.py.
Validation tab expects eval_results.json from evaluate.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

st.set_page_config(page_title="AI Ecosystem Tracker", layout="wide")

COLLECTION = "ai_entities"
# Country / sector / entity-type vocabularies are derived from the data below.
STATUS_COLOUR = {"Verified": "🟢", "Not yet verified": "🟡", "Mismatch found": "🔴"}

OVER_FETCH = 30
MAX_PER_SOURCE = 2

load_dotenv()
API_KEY = os.getenv("MOONSHOT_API_KEY")

SYSTEM_PROMPT = """You are a research assistant answering questions about the AI \
ecosystem of the Southern Mediterranean: Morocco, Algeria, Tunisia, Lebanon and Egypt.

Answer ONLY from the provided context. Do not use outside knowledge. If the context \
only partly answers the question, give what you have and state plainly what is missing. \
Cite the source URL after each claim. If an entry is marked "Not yet verified", say so \
rather than presenting it with full confidence. If two sources disagree about the same \
entity, name the discrepancy rather than silently picking one. This dataset is a \
periodic snapshot: for "latest" questions, cite the most recent dated entry you have \
rather than asserting it is the newest thing that exists. If coverage is uneven across \
countries, say so. Answer in the language of the question. Be concise and factual."""


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading embedding model (slow on first run)...")
def get_embed():
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")


@st.cache_resource
def get_qdrant():
    return qdrant_client.QdrantClient(host="localhost", port=6333)


@st.cache_resource
def get_llm():
    return OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1") if API_KEY else None


@st.cache_data(show_spinner="Loading entities...")
def load_entities():
    client = get_qdrant()
    records, offset = [], None
    while True:
        batch, offset = client.scroll(COLLECTION, limit=256, offset=offset,
                                      with_payload=True, with_vectors=True)
        records.extend(batch)
        if offset is None:
            break
    return pd.DataFrame([{**r.payload, "_id": r.id, "_vector": r.vector} for r in records])


@st.cache_data(show_spinner="Projecting embeddings...")
def project_2d(vectors, method):
    if method == "UMAP":
        try:
            import umap
            return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                             random_state=42).fit_transform(vectors)
        except ImportError:
            st.info("umap-learn not installed; using t-SNE.")
            method = "t-SNE"
    if method == "t-SNE":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, metric="cosine",
                    init="pca", random_state=42).fit_transform(vectors)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(vectors)


def build_filter(country=None, sector=None, etype=None):
    conds = []
    for value, field in ((country, "country"), (sector, "sector"), (etype, "entity_type")):
        if value and value != "All":
            conds.append(FieldCondition(key=field, match=MatchValue(value=value)))
    return Filter(must=conds) if conds else None


def retrieve_diverse(qvec, qf, top_k):
    """Over-fetch then cap per source, so one source can't monopolise results."""
    raw = get_qdrant().query_points(COLLECTION, query=qvec, limit=OVER_FETCH,
                                    with_payload=True, query_filter=qf).points
    counts, kept = {}, []
    for h in raw:
        src = h.payload.get("source_url", "unknown")
        if counts.get(src, 0) >= MAX_PER_SOURCE:
            continue
        counts[src] = counts.get(src, 0) + 1
        kept.append(h)
        if len(kept) >= top_k:
            break
    return kept


def status_badge(status):
    return f"{STATUS_COLOUR.get(status, '⚪')} {status}"


# ---------------------------------------------------------------------- head
st.title("AI Ecosystem Tracker")
st.caption("Morocco · Algeria · Tunisia · Lebanon · Egypt — "
           "grounded, source-verified retrieval over a manually curated dataset")

try:
    df = load_entities()
except Exception as e:
    st.error(f"Cannot reach Qdrant collection '{COLLECTION}'.\n\n"
             f"Check Docker is running and ingest_v3.py has been executed.\n\n{e}")
    st.stop()

# Derived from what is actually in the collection, never hardcoded: a fixed list
# silently produces empty charts whenever the dataset's labels differ from it.
COUNTRIES = sorted(df["country"].dropna().unique().tolist())
SECTORS = sorted(df["sector"].dropna().unique().tolist())
ENTITY_TYPES = sorted(df["entity_type"].dropna().unique().tolist())

tabs = st.tabs(["Search", "Entity", "Compare", "Map", "Overview", "Validation"])

# -------------------------------------------------------------------- SEARCH
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    f_country = c1.selectbox("Country", ["All"] + COUNTRIES, key="s_country")
    f_sector = c2.selectbox("Sector", ["All"] + SECTORS, key="s_sector")
    f_type = c3.selectbox("Entity type", ["All"] + ENTITY_TYPES, key="s_type")

    question = st.text_input("Question",
                             placeholder="Which startups work on medical imaging?")
    top_k = st.slider("Results to retrieve", 3, 15, 8)
    go_btn = st.button("Search", type="primary")

    if go_btn and question.strip():
        qvec = get_embed().get_text_embedding(question)
        hits = retrieve_diverse(qvec, build_filter(f_country, f_sector, f_type), top_k)

        if not hits:
            st.warning("Nothing matches those filters. Try widening them.")
        else:
            llm = get_llm()
            if llm is None:
                st.info("No MOONSHOT_API_KEY in .env — showing retrieved records only.")
            else:
                ctx = "\n\n".join(
                    f"[Source {i}, status: {h.payload['verification_status']}]\n"
                    f"{h.payload['_embedded_text']}"
                    for i, h in enumerate(hits, 1))

                def stream():
                    resp = llm.chat.completions.create(
                        model="kimi-k2.6",
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user",
                                   "content": f"Context:\n{ctx}\n\nQuestion: {question}"}],
                        temperature=1,
                        stream=True)
                    for chunk in resp:
                        d = chunk.choices[0].delta.content
                        if d:
                            yield d

                st.subheader("Answer")
                try:
                    st.write_stream(stream)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

            st.subheader(f"Records retrieved ({len(hits)})")
            for h in hits:
                p = h.payload
                with st.expander(
                        f"{status_badge(p['verification_status'])}  **{p['name']}** — "
                        f"{p['country']} · {p['sector']} · similarity {h.score:.3f}"):
                    st.write(p["description"])
                    m = st.columns(4)
                    m[0].markdown(f"**Type**\n\n{p['entity_type']}")
                    m[1].markdown(f"**Sub-sector**\n\n{p['sub_sector']}")
                    m[2].markdown(f"**Maturity**\n\n{p.get('maturity') or '—'}")
                    m[3].markdown(f"**Funding**\n\n{p.get('funding') or '—'}")
                    if p.get("source_url"):
                        st.markdown(f"[{p.get('source_name', 'Source')}]({p['source_url']})")

# -------------------------------------------------------------------- ENTITY
with tabs[1]:
    st.markdown("Inspect a single record in full, and see which other entities sit "
                "nearest to it in embedding space.")

    ec1, ec2 = st.columns([1, 2])
    filter_country = ec1.selectbox("Filter by country", ["All"] + COUNTRIES, key="e_country")
    pool = df if filter_country == "All" else df[df["country"] == filter_country]
    name = ec2.selectbox(f"Entity ({len(pool)} available)",
                         sorted(pool["name"].unique()), key="e_name")

    rec = pool[pool["name"] == name].iloc[0]

    st.markdown(f"### {rec['name']}")
    st.markdown(f"{status_badge(rec['verification_status'])}")
    st.write(rec["description"])

    g = st.columns(4)
    g[0].metric("Country", rec["country"])
    g[1].metric("Entity type", rec["entity_type"])
    g[2].metric("Sector", rec["sector"])
    g[3].metric("Maturity", rec.get("maturity") or "—")

    with st.expander("All fields"):
        shown = {k: v for k, v in rec.items()
                 if not k.startswith("_") and str(v).strip()}
        st.table(pd.DataFrame({"Field": list(shown), "Value": list(shown.values())}))

    if rec.get("source_url"):
        st.markdown(f"**Source:** [{rec.get('source_name', 'link')}]({rec['source_url']})  "
                    f"· type: {rec.get('source_type', '—')}  "
                    f"· last verified: {rec.get('last_verified', '—')}")

    st.subheader("Nearest entities in embedding space")
    st.caption("Computed from the vectors alone. Nothing told the model these are "
               "related — proximity here means the descriptions mean similar things.")

    neigh = get_qdrant().query_points(COLLECTION, query=list(rec["_vector"]),
                                      limit=7, with_payload=True).points
    rows = [{"Entity": h.payload["name"], "Country": h.payload["country"],
             "Sector": h.payload["sector"], "Type": h.payload["entity_type"],
             "Similarity": round(h.score, 3)}
            for h in neigh if h.payload["name"] != rec["name"]][:6]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- COMPARE
with tabs[2]:
    st.markdown("Compare the catalogued ecosystems of two countries.")

    cc1, cc2 = st.columns(2)
    left = cc1.selectbox("Country A", COUNTRIES, index=3, key="cmp_a")
    right = cc2.selectbox("Country B", COUNTRIES, index=0, key="cmp_b")

    if left == right:
        st.info("Pick two different countries.")
    else:
        dl, dr = df[df["country"] == left], df[df["country"] == right]

        h1, h2 = st.columns(2)
        for col, name_, d in ((h1, left, dl), (h2, right, dr)):
            col.subheader(name_)
            col.metric("Total records", len(d))
            startups = int((d["entity_type"] == "Startup").sum())
            rnd = int((d["entity_type"] == "R&D").sum())
            col.metric("Startups", startups)
            col.metric("Research entities", rnd)
            ratio = f"{startups / rnd:.1f} : 1" if rnd else "n/a"
            col.metric("Startup : research ratio", ratio)
            verified = (d["verification_status"] == "Verified").mean() * 100 if len(d) else 0
            col.metric("Verified", f"{verified:.0f}%")

        st.subheader("Composition by entity type")
        comp = pd.DataFrame({
            left: dl["entity_type"].value_counts(),
            right: dr["entity_type"].value_counts(),
        }).fillna(0).astype(int)
        comp = comp.reindex(sorted(comp.index))
        fig = go.Figure()
        fig.add_bar(y=comp.index, x=comp[left], name=left, orientation="h")
        fig.add_bar(y=comp.index, x=comp[right], name=right, orientation="h")
        fig.update_layout(barmode="group", height=430,
                          xaxis_title="Number of entities", yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Composition by sector")
        sec = (pd.DataFrame({left: dl["sector"].value_counts(),
                             right: dr["sector"].value_counts()})
               .reindex(SECTORS)
               .fillna(0).astype(int))
        fig2 = go.Figure()
        fig2.add_bar(x=sec.index, y=sec[left], name=left)
        fig2.add_bar(x=sec.index, y=sec[right], name=right)
        fig2.update_layout(barmode="group", height=380, yaxis_title="Number of entities")
        st.plotly_chart(fig2, use_container_width=True)

        # sectors present in one and absent in the other
        only_left = sorted(set(dl["sector"]) - set(dr["sector"]))
        only_right = sorted(set(dr["sector"]) - set(dl["sector"]))
        if only_left or only_right:
            st.subheader("Coverage asymmetry")
            if only_left:
                st.write(f"Sectors present in **{left}** but absent in **{right}**: "
                         + ", ".join(only_left))
            if only_right:
                st.write(f"Sectors present in **{right}** but absent in **{left}**: "
                         + ", ".join(only_right))

# ----------------------------------------------------------------------- MAP
with tabs[3]:
    st.markdown("Every record placed by the meaning of its description alone. "
                "Use it as a quality-control tool: the two panels below turn "
                "positions in this space into concrete records worth checking.")

    vectors = np.vstack(df["_vector"].values)
    # cosine similarity on normalised vectors
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    sub1, sub2, sub3 = st.tabs(["Projection", "Possible duplicates",
                                "Possible misclassifications"])

    # ---------------------------------------------------------- projection
    with sub1:
        m1, m2 = st.columns([1, 2])
        method = m1.selectbox("Projection", ["UMAP", "t-SNE", "PCA"])
        colour = m2.selectbox("Colour by",
                              ["sector", "country", "entity_type",
                               "verification_status"])

        coords = project_2d(vectors, method)
        plot_df = df.drop(columns=["_vector"]).copy()
        plot_df["x"], plot_df["y"] = coords[:, 0], coords[:, 1]
        plot_df["short"] = plot_df["description"].str.slice(0, 130) + "..."

        fig = px.scatter(plot_df, x="x", y="y", color=colour, hover_name="name",
                         hover_data={"country": True, "sector": True,
                                     "entity_type": True, "short": True,
                                     "x": False, "y": False},
                         height=640)
        fig.update_traces(marker=dict(size=9, opacity=0.8,
                                      line=dict(width=0.5, color="white")))
        fig.update_layout(xaxis_title=None, yaxis_title=None,
                          legend_title=colour.replace("_", " ").title())
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Axes carry no units; only relative distance means anything. "
                   "Colour by country to see whether entities cluster by what they "
                   "do or by where they are: they cluster by what they do, which is "
                   "why a question about one topic retrieves across all five "
                   "countries at once.")

    # ---------------------------------------------------------- duplicates
    with sub2:
        st.markdown("Pairs of records whose descriptions mean nearly the same thing. "
                    "These are candidate duplicates, or the same entity catalogued "
                    "twice under different names.")
        thresh = st.slider("Similarity threshold", 0.80, 0.99, 0.92, 0.01)

        sims = norm @ norm.T
        np.fill_diagonal(sims, -1)
        iu = np.triu_indices(len(df), k=1)
        pairs = [(i, j, sims[i, j]) for i, j in zip(*iu) if sims[i, j] >= thresh]
        pairs.sort(key=lambda t: -t[2])

        if not pairs:
            st.success(f"No pairs above {thresh:.2f}. Nothing obviously duplicated.")
        else:
            st.warning(f"{len(pairs)} pair(s) above {thresh:.2f}. Review each: "
                       "genuine duplicates should be merged, near-duplicates in "
                       "different countries are usually fine.")
            st.dataframe(pd.DataFrame([{
                "Similarity": round(sc, 3),
                "Record A": df.iloc[i]["name"], "Country A": df.iloc[i]["country"],
                "Record B": df.iloc[j]["name"], "Country B": df.iloc[j]["country"],
                "Same country": df.iloc[i]["country"] == df.iloc[j]["country"],
            } for i, j, sc in pairs[:60]]),
                use_container_width=True, hide_index=True)

    # --------------------------------------------------- misclassifications
    with sub3:
        st.markdown("Records sitting far from the centre of their own sector. "
                    "A record whose description means something unlike the rest of "
                    "its sector is either mislabelled or genuinely unusual. Either "
                    "way it is worth a look.")

        rows = []
        for sector in SECTORS:
            idx = np.where(df["sector"].values == sector)[0]
            if len(idx) < 3:
                continue
            centroid = norm[idx].mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            for i in idx:
                own = float(norm[i] @ centroid)
                # which sector centroid is it actually closest to?
                best, best_s = sector, own
                for other in SECTORS:
                    if other == sector:
                        continue
                    oidx = np.where(df["sector"].values == other)[0]
                    if len(oidx) < 3:
                        continue
                    c = norm[oidx].mean(axis=0)
                    c /= np.linalg.norm(c)
                    sc = float(norm[i] @ c)
                    if sc > best_s:
                        best, best_s = other, sc
                rows.append({
                    "Record": df.iloc[i]["name"],
                    "Country": df.iloc[i]["country"],
                    "Labelled": sector,
                    "Fit to own sector": round(own, 3),
                    "Closest sector": best,
                    "Fit to closest": round(best_s, 3),
                    "Disagrees": best != sector,
                })

        out = pd.DataFrame(rows).sort_values("Fit to own sector")
        disagree = out[out["Disagrees"]]

        c1, c2 = st.columns(2)
        c1.metric("Records checked", len(out))
        c2.metric("Closer to another sector", len(disagree),
                  help="Not necessarily wrong, but these are where the taxonomy "
                       "is doing the least work.")

        st.markdown("**Records closer to a different sector than their own label:**")
        st.dataframe(disagree.drop(columns=["Disagrees"]).head(40),
                     use_container_width=True, hide_index=True)

        st.markdown("**Weakest fit to their own sector (whether or not they disagree):**")
        st.dataframe(out.drop(columns=["Disagrees"]).head(20),
                     use_container_width=True, hide_index=True)

        st.caption("This is an unsupervised cross-check on manual labelling. It "
                   "does not prove a label is wrong, it narrows 521 records down to "
                   "the handful worth re-reading.")

# ------------------------------------------------------------------ OVERVIEW
with tabs[4]:
    a, b, c, d = st.columns(4)
    a.metric("Records", len(df))
    b.metric("Countries", df["country"].nunique())
    c.metric("Verified", f"{(df['verification_status'] == 'Verified').mean() * 100:.0f}%")
    d.metric("Startups", int((df["entity_type"] == "Startup").sum()))

    l, r = st.columns(2)
    l.plotly_chart(px.bar(df["country"].value_counts().sort_values(), orientation="h",
                          title="Records by country",
                          labels={"value": "records", "index": ""}),
                   use_container_width=True)
    r.plotly_chart(px.bar(df["entity_type"].value_counts().sort_values(), orientation="h",
                          title="Records by entity type",
                          labels={"value": "records", "index": ""}),
                   use_container_width=True)

    st.subheader("Coverage matrix")
    st.caption("Thin and empty cells are where the tracker still has gaps. "
               "This doubles as a research plan.")
    matrix = pd.crosstab(df["country"], df["sector"])
    st.plotly_chart(px.imshow(matrix, text_auto=True, aspect="auto",
                              color_continuous_scale="Blues", height=380),
                    use_container_width=True)

    thin = [(cty, sec, int(matrix.loc[cty, sec]))
            for cty in matrix.index for sec in matrix.columns
            if matrix.loc[cty, sec] <= 3]
    if thin:
        st.markdown("**Thinnest cells (3 records or fewer):** "
                    + " · ".join(f"{c}/{s} ({n})" for c, s, n in sorted(thin, key=lambda x: x[2])))

    st.subheader("Browse all records")
    st.dataframe(df.drop(columns=["_vector", "_embedded_text", "_id"], errors="ignore"),
                 use_container_width=True, height=420)

# ---------------------------------------------------------------- VALIDATION
with tabs[5]:
    st.markdown("Retrieval evaluation against a question set whose correct answers "
                "are known in advance. Questions that cannot be scored by string "
                "matching are graded here and counted alongside the automatic ones.")

    res_path = Path("eval_results.json")
    verdict_path = Path("eval_verdicts.json")

    if not res_path.exists():
        st.warning("No eval_results.json found. Run `python evaluate.py` first.")
    else:
        ev = json.loads(res_path.read_text(encoding="utf-8"))
        results = ev["results"]

        verdicts = {}
        if verdict_path.exists():
            verdicts = json.loads(verdict_path.read_text(encoding="utf-8"))

        auto = [r for r in results if r["scoring"] == "automatic"]
        manual = [r for r in results if r["scoring"] == "manual"]

        graded = [r for r in manual if verdicts.get(r["id"], {}).get("verdict")]
        passed = [r for r in graded
                  if verdicts[r["id"]]["verdict"] == "Pass"]

        # ---------------------------------------------------------- headline
        m = st.columns(5)
        m[0].metric(f"Recall@{ev['top_k']}", f"{ev['average_recall']:.0%}",
                    help=f"Automatic questions only ({len(auto)} of {len(results)}).")
        m[1].metric("Fully correct", f"{ev['fully_correct']}/{len(auto)}")
        m[2].metric("Manual graded", f"{len(graded)}/{len(manual)}")
        m[3].metric("Manual passed",
                    f"{len(passed)}/{len(graded)}" if graded else "—")
        ins = ev.get("mean_top_score_in_scope")
        oos = ev.get("mean_top_score_out_of_scope")
        if ins is not None and oos is not None:
            m[4].metric("Score separation", f"{ins - oos:+.3f}",
                        help="Mean top similarity for in-scope questions minus "
                             "out-of-scope. Positive means retrieval scores carry "
                             "signal about whether the dataset covers a question.")

        overall_n = len(auto) + len(graded)
        overall_pass = ev["fully_correct"] + len(passed)
        if overall_n:
            st.success(f"**Overall: {overall_pass}/{overall_n} questions pass** "
                       f"({overall_pass / overall_n:.0%}), combining automatic recall "
                       f"and graded manual review."
                       + (f" {len(manual) - len(graded)} manual question(s) still "
                          f"ungraded." if len(graded) < len(manual) else ""))

        # ------------------------------------------------------------- charts
        cA, cB = st.columns([3, 2])
        rec_df = pd.DataFrame([{"id": r["id"], "category": r["category"],
                                "question": r["question"], "recall": r["recall"]}
                               for r in auto])
        fig = px.bar(rec_df, x="id", y="recall", color="category",
                     hover_data=["question"], range_y=[0, 1.05], height=360,
                     labels={"recall": f"Recall@{ev['top_k']}", "id": ""},
                     title="Automatic questions")
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
        cA.plotly_chart(fig, use_container_width=True)

        if ins is not None and oos is not None:
            cB.plotly_chart(
                px.bar(x=["In-scope", "Out-of-scope"], y=[ins, oos], height=360,
                       labels={"x": "", "y": "Mean top similarity"},
                       title="Does the system know what it doesn't know?"),
                use_container_width=True)

        # ------------------------------------------------------ manual grading
        st.subheader("Manual review")
        st.caption("These test properties that string matching cannot check: "
                   "refusal on unanswerable questions, cross-lingual retrieval, "
                   "verification awareness, comparative answers. Grade each one, "
                   "then export the table for the report.")

        for r in manual:
            saved = verdicts.get(r["id"], {})
            current = saved.get("verdict")
            icon = {"Pass": "✅", "Fail": "❌", "Partial": "🟡"}.get(current, "⬜")

            with st.expander(f"{icon} [{r['id']}] ({r['category']}) {r['question']}",
                             expanded=current is None):
                st.markdown(f"**Pass condition:** {r.get('notes') or '—'}")

                if r["retrieved"]:
                    st.markdown("**Retrieved records:**")
                    st.dataframe(
                        pd.DataFrame({
                            "Entity": r["retrieved"],
                            "Country": r.get("retrieved_countries", []),
                            "Score": r["scores"],
                        }),
                        use_container_width=True, hide_index=True)
                else:
                    st.info("Nothing retrieved.")

                if r["category"] == "out_of_scope":
                    st.caption("Run this question in the Search tab and read the "
                               "generated answer. Pass = the system says it has no "
                               "data. Fail = it answers anyway.")

                g1, g2 = st.columns([1, 3])
                choice = g1.radio("Verdict", ["Ungraded", "Pass", "Partial", "Fail"],
                                  index=["Ungraded", "Pass", "Partial", "Fail"].index(
                                      current) if current else 0,
                                  key=f"v_{r['id']}")
                note = g2.text_input("Note (what you observed)",
                                     value=saved.get("note", ""),
                                     key=f"n_{r['id']}")

                if st.button("Save verdict", key=f"b_{r['id']}"):
                    verdicts[r["id"]] = {
                        "verdict": None if choice == "Ungraded" else choice,
                        "note": note,
                    }
                    verdict_path.write_text(
                        json.dumps(verdicts, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    st.success("Saved.")
                    st.rerun()

        # ------------------------------------------------- automatic detail
        st.subheader("Automatic question detail")
        for r in auto:
            icon = "✅" if r["recall"] == 1.0 else ("🟡" if r["recall"] > 0 else "❌")
            with st.expander(f"{icon} [{r['id']}] {r['question']} "
                             f"— recall {r['recall']:.2f}"):
                st.write(f"**Expected:** {', '.join(r['expected_names'])}")
                st.write(f"**Retrieved:** {', '.join(r['retrieved'])}")
                st.write(f"**Scores:** {r['scores']}")
                if r.get("missing"):
                    st.warning(f"Missing: {', '.join(r['missing'])}")
                if r.get("notes"):
                    st.caption(r["notes"])

        # -------------------------------------------------------- export
        st.subheader("Export for the report")

        rows = []
        for r in results:
            if r["scoring"] == "automatic":
                outcome = ("Pass" if r["recall"] == 1.0
                           else ("Partial" if r["recall"] > 0 else "Fail"))
                score = f"{r['recall']:.2f}"
            else:
                v = verdicts.get(r["id"], {}).get("verdict")
                outcome = v or "Ungraded"
                score = "manual"
            rows.append({"ID": r["id"], "Category": r["category"].replace("_", " "),
                         "Question": r["question"], "Scoring": r["scoring"],
                         "Recall": score, "Outcome": outcome})
        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True, hide_index=True)

        e1, e2 = st.columns(2)
        e1.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                           "validation_results.csv", "text/csv")

        def latex_table(t):
            head_ = ("\\begin{table}[h]\n\\centering\\small\n"
                     "\\caption{Retrieval validation results}\n"
                     "\\begin{tabular}{@{}llp{6.4cm}ll@{}}\n\\toprule\n"
                     "\\textbf{ID} & \\textbf{Category} & \\textbf{Question} & "
                     "\\textbf{Scoring} & \\textbf{Outcome} \\\\\n\\midrule\n")
            body = ""
            for _, x in t.iterrows():
                q = (str(x["Question"]).replace("&", "\\&").replace("%", "\\%")
                     .replace("_", "\\_"))
                body += (f"{x['ID']} & {x['Category']} & {q} & "
                         f"{x['Scoring']} & {x['Outcome']} \\\\\n")
            return head_ + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

        e2.download_button("Download LaTeX table", latex_table(table).encode("utf-8"),
                           "validation_table.tex", "text/plain")

        with st.expander("Summary paragraph for the report"):
            sep = (f"{ins - oos:+.3f}" if ins is not None and oos is not None else "n/a")
            st.code(
                f"The system was evaluated against {len(results)} questions, of which "
                f"{len(auto)} carry known correct answers and were scored automatically "
                f"and {len(manual)} were graded manually. Average Recall@{ev['top_k']} "
                f"across the automatically scored questions was "
                f"{ev['average_recall']:.0%}, with {ev['fully_correct']} of {len(auto)} "
                f"returning every expected record. Of the {len(graded)} manually graded "
                f"questions, {len(passed)} passed. Mean top similarity was "
                f"{ins:.3f} for in-scope questions against {oos:.3f} for deliberately "
                f"out-of-scope questions, a separation of {sep}, indicating that "
                f"retrieval scores carry usable signal about whether the dataset covers "
                f"a given question at all."
                if ins is not None and oos is not None else
                f"The system was evaluated against {len(results)} questions.",
                language=None)
