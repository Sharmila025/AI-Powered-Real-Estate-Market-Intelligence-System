# Parcl – Buyer Segmentation & Investment Profiling

## Run
pip install -r requirements.txt
python pipeline.py          # cleans data, clusters, writes data/*.csv, data/metrics.json, figures/*.png  (--k 3 to change k)
streamlit run app.py        # dashboard

## Files
- pipeline.py  – cleaning, feature engineering (joins purchase behaviour from properties.csv), encoding, scaling, K-Means + Ward hierarchical, elbow/silhouette, profiling
- app.py       – Streamlit dashboard (overview, investor behaviour, geography, segment insights, model quality) with country/region/purpose/client-type filters
- data/        – input CSVs and generated outputs (clustered_clients.csv, cluster_profile.csv, k_evaluation.csv, metrics.json)
- figures/     – k_selection, dendrogram, pca_segments, eda_overview, segment_heatmap
