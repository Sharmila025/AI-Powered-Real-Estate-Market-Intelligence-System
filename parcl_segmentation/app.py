"""Parcl – Buyer Segmentation & Investment Profiling dashboard.
Run:  streamlit run app.py
Requires data/clustered_clients.csv (created by `python pipeline.py`).
"""
import json, os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

st.set_page_config(page_title="Parcl Buyer Intelligence", page_icon="🏙️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 8% 0%, #243b70 0, #101827 34%, #0b1120 72%);
        color: #edf4ff;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(11, 17, 32, .72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17254a 0%, #111a31 58%, #0d1426 100%);
        border-right: 1px solid rgba(130, 173, 255, .22);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label { color: #dce9ff; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -.03em; }
    h1 { font-size: clamp(2rem, 4vw, 3.4rem) !important; margin-bottom: .15rem; }
    .hero {
        padding: 1.4rem 1.7rem 1.25rem;
        margin: .4rem 0 1.2rem;
        border-radius: 24px;
        background: linear-gradient(120deg, rgba(76, 113, 255, .38), rgba(20, 184, 166, .2));
        border: 1px solid rgba(151, 191, 255, .3);
        box-shadow: 0 18px 50px rgba(0, 0, 0, .22);
    }
    .hero-kicker { color: #93c5fd; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero-title { color: #fff; font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; margin: .25rem 0; }
    .hero-copy { color: #c9d8f5; margin: 0; }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(147, 197, 253, .2);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(37, 61, 111, .75), rgba(20, 31, 60, .72));
        box-shadow: 0 10px 24px rgba(0, 0, 0, .14);
    }
    [data-testid="stMetricLabel"] { color: #a9c4ed; }
    [data-testid="stMetricValue"] { color: #f8fbff; }
    .stTabs [data-baseweb="tab-list"] { gap: .45rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .55rem .85rem; }
    .stTabs [aria-selected="true"] { background: #3156c7; color: white; }
    div[data-testid="stAlert"] { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

US_ABBR = {"Alabama": "AL", "Arizona": "AZ", "California": "CA", "Colorado": "CO", "Florida": "FL",
           "Georgia": "GA", "Kansas": "KS", "Nevada": "NV", "New York": "NY", "Ohio": "OH",
           "Oregon": "OR", "Texas": "TX", "Utah": "UT", "Virginia": "VA", "Washington": "WA",
           "Wyoming": "WY"}
ISO3 = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU", "France": "FRA", "Belgium": "BEL",
        "Mexico": "MEX", "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}


@st.cache_data
def load():
    required = ["clustered_clients.csv", "cluster_profile.csv", "metrics.json", "k_evaluation.csv"]
    missing = [name for name in required if not Path(D, name).exists()]

    # A fresh Streamlit deployment may contain the raw inputs but not generated outputs.
    # Build those outputs once at startup so the dashboard remains self-contained.
    if missing and Path(D, "clients.csv").exists() and Path(D, "properties.csv").exists():
        from pipeline import main as build_pipeline
        build_pipeline(Path(D, "clients.csv"), Path(D, "properties.csv"), k=4)
        missing = [name for name in required if not Path(D, name).exists()]

    if missing:
        expected = ", ".join(required)
        st.error(
            "Dashboard data is missing from the deployment. Upload the complete `data` folder "
            f"with `{expected}`, or include `data/clients.csv` and `data/properties.csv` "
            "so the dashboard can generate the outputs automatically."
        )
        st.stop()

    df = pd.read_csv(Path(D, "clustered_clients.csv"))
    prof = pd.read_csv(Path(D, "cluster_profile.csv"))
    with open(Path(D, "metrics.json"), encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    ev = pd.read_csv(Path(D, "k_evaluation.csv"))
    return df, prof, metrics, ev


df, prof, metrics, ev = load()
SEGS = sorted(df.segment.unique())
PALETTE = px.colors.qualitative.Set2
CMAP = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SEGS)}

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("Filters")
countries = st.sidebar.multiselect("Country", sorted(df.country.unique()))
regions_pool = df[df.country.isin(countries)] if countries else df
regions = st.sidebar.multiselect("Region", sorted(regions_pool.region.unique()))
purpose = st.sidebar.multiselect("Acquisition purpose", sorted(df.acquisition_purpose.unique()))
ctype = st.sidebar.multiselect("Client type", sorted(df.client_type.unique()))

f = df.copy()
if countries: f = f[f.country.isin(countries)]
if regions: f = f[f.region.isin(regions)]
if purpose: f = f[f.acquisition_purpose.isin(purpose)]
if ctype: f = f[f.client_type.isin(ctype)]

st.sidebar.markdown(f"**{len(f):,}** of {len(df):,} clients selected")
if f.empty:
    st.warning("No clients match these filters.")
    st.stop()

st.markdown("""
<section class="hero">
  <div class="hero-kicker">Parcl intelligence studio</div>
  <div class="hero-title">🏙️ Buyer Segmentation &amp; Investment Profiling</div>
  <p class="hero-copy">Explore 2,000 clients through interactive segments, investment behavior, geography, and model quality.</p>
</section>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Segmentation overview", "Investor behaviour", "Geographic analysis", "Segment insights", "Model quality"])

# ------------------------------------------------------------------ 1. overview
with tab1:
    c = st.columns(4)
    c[0].metric("Clients", f"{len(f):,}")
    c[1].metric("Avg. spend / client", f"${f.total_spend.mean():,.0f}")
    c[2].metric("Investment share", f"{(f.acquisition_purpose == 'Investment').mean():.0%}")
    c[3].metric("Loan share", f"{(f.loan_applied == 'Yes').mean():.0%}")

    cnt = f.segment.value_counts().rename_axis("segment").reset_index(name="clients")
    l, r = st.columns(2)
    l.plotly_chart(px.pie(cnt, names="segment", values="clients", hole=.45, color="segment",
                          color_discrete_map=CMAP, title="Cluster distribution"))
    r.plotly_chart(px.bar(cnt, x="segment", y="clients", color="segment", color_discrete_map=CMAP,
                          text="clients", title="Clients per segment").update_layout(showlegend=False),
                   )
    st.plotly_chart(px.scatter(f, x="pc1", y="pc2", color="segment", color_discrete_map=CMAP, opacity=.6,
                               hover_data=["client_id", "country", "acquisition_purpose"],
                               title="Client map (PCA projection of the clustering feature space)"),
                    )

# ------------------------------------------------------------------ 2. investor behaviour
with tab2:
    l, r = st.columns(2)
    ap = (f.groupby(["segment", "acquisition_purpose"]).size().reset_index(name="n"))
    ap["share"] = ap.n / ap.groupby("segment").n.transform("sum")
    l.plotly_chart(px.bar(ap, x="segment", y="share", color="acquisition_purpose", barmode="stack",
                          title="Acquisition purpose by segment", labels={"share": "share of clients"}),
                   )
    ln = f.groupby(["segment", "loan_applied"]).size().reset_index(name="n")
    ln["share"] = ln.n / ln.groupby("segment").n.transform("sum")
    r.plotly_chart(px.bar(ln, x="segment", y="share", color="loan_applied", barmode="stack",
                          title="Financing (loan applied) by segment", labels={"share": "share of clients"}),
                   )
    l, r = st.columns(2)
    l.plotly_chart(px.box(f, x="segment", y="total_spend", color="segment", color_discrete_map=CMAP,
                          title="Total spend per client").update_layout(showlegend=False))
    r.plotly_chart(px.box(f, x="segment", y="units_purchased", color="segment", color_discrete_map=CMAP,
                          title="Units purchased per client").update_layout(showlegend=False))
    rc = f.groupby(["segment", "referral_channel"]).size().reset_index(name="n")
    rc["share"] = rc.n / rc.groupby("segment").n.transform("sum")
    st.plotly_chart(px.bar(rc, x="segment", y="share", color="referral_channel", barmode="group",
                           title="Acquisition channel by segment"))

# ------------------------------------------------------------------ 3. geography
with tab3:
    metric = st.radio("Map shows", ["Dominant segment", "Client count", "Avg. spend"], horizontal=True)
    g = f.groupby("country").agg(clients=("client_id", "size"), spend=("total_spend", "mean"),
                                 dom=("segment", lambda s: s.mode().iloc[0])).reset_index()
    g["iso"] = g.country.map(ISO3)
    if metric == "Dominant segment":
        fig = px.choropleth(g, locations="iso", color="dom", color_discrete_map=CMAP, hover_name="country",
                            hover_data=["clients"], title="Dominant buyer segment by country")
    else:
        col = "clients" if metric == "Client count" else "spend"
        fig = px.choropleth(g, locations="iso", color=col, hover_name="country", color_continuous_scale="Blues",
                            title=f"{metric} by country")
    st.plotly_chart(fig)

    us = f[f.country == "USA"].copy()
    if not us.empty:
        us["state"] = us.region.map(US_ABBR)
        st_g = us.groupby("state").agg(clients=("client_id", "size"),
                                       dom=("segment", lambda s: s.mode().iloc[0]),
                                       inv=("acquisition_purpose", lambda s: (s == "Investment").mean())).reset_index()
        l, r = st.columns(2)
        l.plotly_chart(px.choropleth(st_g, locations="state", locationmode="USA-states", scope="usa",
                                     color="clients", color_continuous_scale="Blues",
                                     title="US clients by state"))
        r.plotly_chart(px.choropleth(st_g, locations="state", locationmode="USA-states", scope="usa",
                                     color="inv", color_continuous_scale="Oranges",
                                     title="Investment share by state", labels={"inv": "investment share"}),
                       )
    reg = f.groupby(["region", "segment"]).size().reset_index(name="n")
    top = f.region.value_counts().head(15).index
    st.plotly_chart(px.bar(reg[reg.region.isin(top)], x="region", y="n", color="segment",
                           color_discrete_map=CMAP, title="Segment mix – top 15 regions"),
                    )

# ------------------------------------------------------------------ 4. insights
with tab4:
    st.subheader("Segment profiles (filtered clients)")
    summ = f.groupby("segment").agg(
        clients=("client_id", "size"), avg_age=("age", "mean"), satisfaction=("satisfaction_score", "mean"),
        units=("units_purchased", "mean"), total_spend=("total_spend", "mean"), avg_price=("avg_price", "mean"),
        avg_area_sqft=("avg_area_sqft", "mean"),
        investment_share=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        loan_share=("loan_applied", lambda s: (s == "Yes").mean()),
        company_share=("client_type", lambda s: (s == "Company").mean()),
        international_share=("country", lambda s: (s != "USA").mean())).round(2)
    st.dataframe(summ.style.format({"investment_share": "{:.0%}", "loan_share": "{:.0%}",
                                    "company_share": "{:.0%}", "international_share": "{:.0%}",
                                    "total_spend": "${:,.0f}", "avg_price": "${:,.0f}"}),
                 )
    ind = summ[["avg_age", "satisfaction", "units", "avg_price", "investment_share", "loan_share",
                "international_share"]]
    z = (ind - ind.mean()) / ind.std(ddof=0).replace(0, 1)
    st.plotly_chart(px.imshow(z.round(2), color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                              title="Segment traits relative to the average segment (z-score)"),
                    )
    pick = st.selectbox("Inspect a segment", SEGS)
    st.dataframe(f[f.segment == pick].drop(columns=["pc1", "pc2", "cluster", "hier_cluster"]).head(200),
                 )
    st.download_button("Download filtered clients (CSV)", f.to_csv(index=False), "filtered_clients.csv")

# ------------------------------------------------------------------ 5. model quality
with tab5:
    c = st.columns(4)
    c[0].metric("Clusters used", metrics["k"])
    c[1].metric("Silhouette (K-Means)", f"{metrics['silhouette_kmeans']:.3f}")
    c[2].metric("Silhouette (Hierarchical)", f"{metrics['silhouette_hier']:.3f}")
    c[3].metric("K-Means vs Hierarchical ARI", f"{metrics['ari_kmeans_vs_hier']:.2f}")
    l, r = st.columns(2)
    l.plotly_chart(px.line(ev, x="k", y="inertia", markers=True, title="Elbow method"))
    r.plotly_chart(px.line(ev, x="k", y=["sil_kmeans", "sil_hier"], markers=True, title="Silhouette score"),
                   )
    st.info(f"Silhouette is highest at k={metrics['silhouette_optimal_k']}, but k={metrics['k']} was used because it "
            "separates the small multi-unit 'Portfolio Builders' group and splits the main base by satisfaction. "
            "Silhouette values below ~0.25 indicate overlapping clusters: segments are useful for targeting, "
            "not sharply separated populations.")
