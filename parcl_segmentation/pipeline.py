"""
Parcl buyer segmentation pipeline.
Steps: clean -> join purchase behaviour -> encode -> scale -> K-Means + Hierarchical -> elbow/silhouette -> cluster profiling -> export for dashboard + paper.

Usage:  python pipeline.py --clients clients.csv --properties properties.csv
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DATA = os.path.join(HERE, "data")
OUT_FIG = os.path.join(HERE, "figures")
REF_YEAR = 2026
SEED = 42
GEO_WEIGHT = 0.5
DEFAULT_K = 4          # PRD asks for four buyer segments; silhouette-optimal k is reported alongside


# Step 1: cleaning
def clean_clients(path):
    c = pd.read_csv(path)
    n0 = len(c)
    for col in c.select_dtypes(include=["object", "string"]).columns:
        c[col] = c[col].astype(str).str.strip()
    c = c.replace({"nan": np.nan, "": np.nan})
    # normalise labels
    c["client_type"] = c["client_type"].str.title().replace({"Corporate": "Company"})
    c["gender"] = c["gender"].str.upper().str[0]
    c["acquisition_purpose"] = c["acquisition_purpose"].str.title()
    c["loan_applied"] = c["loan_applied"].str.title()
    c["referral_channel"] = c["referral_channel"].str.title()
    c["country"] = c["country"].str.strip()
    c["region"] = c["region"].str.strip()
    # duplicates
    c = c.drop_duplicates(subset="client_id").drop_duplicates(
        subset=["first_name", "last_name", "date_of_birth", "country"])
    # DOB uses mixed formats (MM/DD/YYYY and ambiguous NN-NN-YYYY) -> only the year is reliable
    c["birth_year"] = pd.to_numeric(c["date_of_birth"].str[-4:], errors="coerce")
    c["age"] = REF_YEAR - c["birth_year"]
    # missing values: median for numeric, mode for categorical
    for col in ["age", "satisfaction_score"]:
        c[col] = c[col].fillna(c[col].median())
    for col in ["client_type", "gender", "country", "region", "acquisition_purpose",
                "loan_applied", "referral_channel"]:
        c[col] = c[col].fillna(c[col].mode().iloc[0])
    print(f"[clean] clients {n0} -> {len(c)} rows")
    return c


def purchase_features(path):
    p = pd.read_csv(path)
    p["sale_price"] = p["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
    p["transaction_date"] = pd.to_datetime(p["transaction_date"], format="%m-%d-%Y", errors="coerce")
    sold = p[(p["listing_status"].str.title() == "Sold") & p["client_ref"].notna()].copy()
    g = sold.groupby("client_ref")
    f = pd.DataFrame({
        "units_purchased": g.size(),
        "total_spend": g["sale_price"].sum(),
        "avg_price": g["sale_price"].mean(),
        "avg_area_sqft": g["floor_area_sqft"].mean(),
        "office_share": g["unit_category"].apply(lambda s: (s == "Office").mean()),
        "first_purchase": g["transaction_date"].min(),
    }).reset_index().rename(columns={"client_ref": "client_id"})
    f["price_per_sqft"] = f["total_spend"] / g["floor_area_sqft"].sum().values
    return f, p


#  Steps 2-3: encode + scale
NUM_COLS = ["age", "satisfaction_score", "units_purchased", "total_spend", "avg_price", "avg_area_sqft"]
OHE_COLS = ["client_type", "region_group", "acquisition_purpose", "referral_channel", "country_group"]


def build_matrix(df):
    d = df.copy()
    # Label encoding for binary flags
    d["loan_applied_enc"] = (d["loan_applied"] == "Yes").astype(int)
    d["gender_enc"] = (d["gender"] == "F").astype(int)
    d["intl_enc"] = (d["country"] != "USA").astype(int)
    # collapse rare regions / countries so one-hot stays interpretable
    top_r = d["region"].value_counts().head(8).index
    d["region_group"] = np.where(d["region"].isin(top_r), d["region"], "Other")
    d["country_group"] = np.where(d["country"].isin(["USA", "UK", "Canada"]), d["country"], "Other")
    ohe = pd.get_dummies(d[OHE_COLS], dtype=int)
    # log-transform skewed money fields before scaling
    num = d[NUM_COLS].copy()
    for col in ["total_spend", "avg_price"]:
        num[col] = np.log1p(num[col])
    scaled = pd.DataFrame(StandardScaler().fit_transform(num), columns=NUM_COLS, index=d.index)
    # geography one-hots (many sparse columns) are down-weighted so they do not dominate distances;
    # gender is kept for profiling but excluded from clustering (not a marketing-relevant driver)
    geo_cols = [c for c in ohe.columns if c.startswith(("region_group", "country_group"))]
    ohe[geo_cols] = ohe[geo_cols] * GEO_WEIGHT
    X = pd.concat([scaled, d[["loan_applied_enc", "intl_enc"]], ohe], axis=1)
    return X, d


# Steps 4-5: models
def evaluate_k(X, ks=range(2, 11)):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        rows.append(dict(k=k, inertia=km.inertia_,
                         sil_kmeans=silhouette_score(X, km.labels_),
                         dbi_kmeans=davies_bouldin_score(X, km.labels_),
                         sil_hier=silhouette_score(X, hc.labels_)))
    return pd.DataFrame(rows)


def plot_k_selection(ev, k):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ev.k, ev.inertia, "o-", color="#1f4e79")
    ax[0].axvline(k, ls="--", color="grey"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia (WCSS)")
    ax[1].plot(ev.k, ev.sil_kmeans, "o-", label="K-Means", color="#1f4e79")
    ax[1].plot(ev.k, ev.sil_hier, "s--", label="Hierarchical (Ward)", color="#c55a11")
    ax[1].axvline(k, ls="--", color="grey"); ax[1].legend()
    ax[1].set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    plt.tight_layout(); plt.savefig(f"{OUT_FIG}/k_selection.png", dpi=160); plt.close()


#  Step 6: naming
def name_clusters(prof):
    """Name clusters from their measured traits (not forced onto the PRD labels)."""
    names, left = {}, list(prof.index)
    top_units = prof.loc[left, "units_purchased"].idxmax()
    names[top_units] = "Portfolio Builders"; left.remove(top_units)
    prem = prof.loc[left, "avg_price"].idxmax()
    names[prem] = "Premium Buyers"; left.remove(prem)
    entry = prof.loc[left, "avg_price"].idxmin()
    names[entry] = "Entry-Level Buyers"; left.remove(entry)
    for cid in left:
        names[cid] = "Satisfied Core Buyers" if prof.loc[cid, "satisfaction"] >= prof["satisfaction"].median() else f"Segment {cid + 1}"
    return names


def main(clients, properties, k=None):
    os.makedirs(OUT_DATA, exist_ok=True); os.makedirs(OUT_FIG, exist_ok=True)
    c = clean_clients(clients)
    f, props = purchase_features(properties)
    df = c.merge(f, on="client_id", how="left")
    df["units_purchased"] = df["units_purchased"].fillna(0)
    for col in ["total_spend", "avg_price", "avg_area_sqft", "office_share", "price_per_sqft"]:
        df[col] = df[col].fillna(0)
    X, d = build_matrix(df)

    ev = evaluate_k(X)
    # choose k: best silhouette among k>=3 so the segmentation stays actionable
    best_k = int(ev[ev.k >= 3].sort_values("sil_kmeans", ascending=False).iloc[0].k)
    k = k or DEFAULT_K
    plot_k_selection(ev, k)
    print(ev.round(3).to_string(index=False)); print("k used =", k, "| silhouette-optimal k =", best_k)

    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    d["cluster"] = km.labels_
    d["hier_cluster"] = hc.labels_
    ari = adjusted_rand_score(km.labels_, hc.labels_)

    # dendrogram (truncated)
    Z = linkage(X.values, method="ward")
    plt.figure(figsize=(10, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=24, no_labels=True, color_threshold=Z[-(k - 1), 2])
    plt.title("Hierarchical clustering (Ward) – truncated dendrogram"); plt.ylabel("Distance")
    plt.tight_layout(); plt.savefig(f"{OUT_FIG}/dendrogram.png", dpi=160); plt.close()

    # profile
    d["is_company"] = (d.client_type == "Company").astype(int)
    prof = d.groupby("cluster").agg(
        size=("client_id", "size"), age=("age", "mean"), satisfaction=("satisfaction_score", "mean"),
        units_purchased=("units_purchased", "mean"), total_spend=("total_spend", "mean"),
        avg_price=("avg_price", "mean"), avg_area_sqft=("avg_area_sqft", "mean"),
        office_share=("office_share", "mean"),
        loan_share=("loan_applied_enc", "mean"), intl_share=("intl_enc", "mean"),
        company_share=("is_company", "mean"),
        investment_share=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        home_share=("acquisition_purpose", lambda s: (s == "Home").mean()),
    )
    names = name_clusters(prof)
    d["segment"] = d["cluster"].map(names)
    prof["segment"] = prof.index.map(names)
    prof.round(3).to_csv(f"{OUT_DATA}/cluster_profile.csv")

    # PCA map
    pcs = PCA(n_components=2, random_state=SEED).fit(X)
    xy = pcs.transform(X); d["pc1"], d["pc2"] = xy[:, 0], xy[:, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg, g in d.groupby("segment"):
        ax.scatter(g.pc1, g.pc2, s=10, alpha=.6, label=seg)
    ax.legend(); ax.set(title=f"Segments in PCA space ({pcs.explained_variance_ratio_.sum():.0%} variance)",
                        xlabel="PC1", ylabel="PC2")
    plt.tight_layout(); plt.savefig(f"{OUT_FIG}/pca_segments.png", dpi=160); plt.close()

    keep = ["client_id", "first_name", "last_name", "client_type", "gender", "country", "region",
            "acquisition_purpose", "loan_applied", "referral_channel", "satisfaction_score", "age",
            "units_purchased", "total_spend", "avg_price", "avg_area_sqft", "office_share",
            "cluster", "segment", "hier_cluster", "pc1", "pc2"]
    d[keep].to_csv(f"{OUT_DATA}/clustered_clients.csv", index=False)
    props.to_csv(f"{OUT_DATA}/properties_clean.csv", index=False)
    metrics = dict(k=k, silhouette_optimal_k=best_k, silhouette_kmeans=float(silhouette_score(X, km.labels_)),
                   silhouette_hier=float(silhouette_score(X, hc.labels_)),
                   davies_bouldin=float(davies_bouldin_score(X, km.labels_)),
                   ari_kmeans_vs_hier=float(ari), n_features=int(X.shape[1]),
                   pca_var=float(pcs.explained_variance_ratio_.sum()), names={int(k_): v for k_, v in names.items()})
    json.dump(metrics, open(f"{OUT_DATA}/metrics.json", "w"), indent=2)
    ev.to_csv(f"{OUT_DATA}/k_evaluation.csv", index=False)
    print(json.dumps(metrics, indent=2)); print(prof.round(2).T.to_string())
    return d, prof, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", default=os.path.join(HERE, "data", "clients.csv"))
    ap.add_argument("--properties", default=os.path.join(HERE, "data", "properties.csv"))
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    main(a.clients, a.properties, a.k)
