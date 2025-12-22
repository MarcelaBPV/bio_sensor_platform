# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma Integrada
Raman + Questionário + Otimizador (ML)
⚠ Uso em pesquisa. NÃO é diagnóstico médico.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import uuid

import raman_processing as rp
from ml_otimizador import train_random_forest_from_features, MLConfig

# =========================================================
# CONFIGURAÇÃO
# =========================================================
st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Plataforma Integrada")

# =========================================================
# SESSION STATE (CONTROLADO)
# =========================================================
state_defaults = {
    "raman_results": None,
    "questionnaire_df": None,
    "ml_dataset": pd.DataFrame(),
    "stats_ready": False,
}

for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Parâmetros Raman")
    fit_model = st.selectbox("Ajuste de pico", [None, "gauss", "lorentz", "voigt"])
    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima", 1, 500, 5)

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Raman", "Questionário", "Otimizador (ML + Estatística)"])

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    sample_file = st.file_uploader(
        "Upload do espectro Raman",
        type=["txt", "csv", "xls", "xlsx"],
    )

    if st.button("▶ Processar espectro"):
        if sample_file:
            st.session_state.raman_results = rp.process_raman_spectrum_with_groups(
                sample_file,
                peak_height=peak_height,
                peak_distance=peak_distance,
                peak_prominence=peak_prominence,
                fit_model=fit_model,
            )
            st.success("Espectro processado.")
        else:
            st.warning("Faça upload de um espectro.")

    if st.session_state.raman_results:
        data = st.session_state.raman_results
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"])
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])

    if q_file:
        st.session_state.questionnaire_df = pd.read_csv(q_file)
        st.success("Questionário carregado.")

    if st.session_state.questionnaire_df is not None:
        st.dataframe(
            st.session_state.questionnaire_df.head(),
            use_container_width=True,
        )

# =========================================================
# ABA 3 — OTIMIZADOR (ML + ESTATÍSTICA)
# =========================================================
with tab3:
    st.header("Análise Estatística Raman × Questionário")

    if st.session_state.ml_dataset.empty:
        st.info("Dataset ainda vazio. Processe espectros e salve features.")
        st.stop()

    df = st.session_state.ml_dataset.copy()

    # -----------------------------
    # SEPARAÇÃO FEATURES / META
    # -----------------------------
    meta_cols = [c for c in df.columns if c in ["genero", "fumante", "doenca", "label"]]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].fillna(0.0)

    # -----------------------------
    # NORMALIZAÇÃO (PADRÃO ARTIGO)
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # PCA
    # -----------------------------
    st.subheader("Análise de Componentes Principais (PCA)")

    n_components = st.slider(
        "Número de componentes principais",
        min_value=2,
        max_value=min(10, X.shape[1]),
        value=2,
    )

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    st.write(
        "Variância explicada acumulada:",
        np.cumsum(pca.explained_variance_ratio_),
    )

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    st.subheader("Clustering não supervisionado")

    k = st.slider("Número de clusters (k)", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    df["cluster"] = clusters

    # -----------------------------
    # PLOT PCA
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters,
        cmap="tab10",
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA + KMeans (dados Raman)")
    plt.colorbar(sc, ax=ax, label="Cluster")
    st.pyplot(fig)

    # -----------------------------
    # ESTATÍSTICA POR CLUSTER
    # -----------------------------
    st.subheader("Distribuição estatística por cluster")

    if "genero" in df.columns:
        st.write("Gênero × Cluster")
        st.dataframe(pd.crosstab(df["cluster"], df["genero"]))

    if "fumante" in df.columns:
        st.write("Fumante × Cluster")
        st.dataframe(pd.crosstab(df["cluster"], df["fumante"]))

    if "doenca" in df.columns:
        st.write("Doença × Cluster")
        st.dataframe(pd.crosstab(df["cluster"], df["doenca"]))
