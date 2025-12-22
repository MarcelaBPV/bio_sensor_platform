# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma Integrada
Raman + Questionário + Otimizador Estatístico (PCA + Clustering)

⚠ Uso exclusivo em pesquisa. NÃO é diagnóstico médico.
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import raman_processing as rp

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Plataforma Integrada")

# =========================================================
# SESSION STATE (CONTROLADO, SEM LOOP)
# =========================================================
if "raman_results" not in st.session_state:
    st.session_state.raman_results = None

if "questionnaire_df" not in st.session_state:
    st.session_state.questionnaire_df = None

if "ml_dataset" not in st.session_state:
    st.session_state.ml_dataset = pd.DataFrame()

# =========================================================
# SIDEBAR — PARÂMETROS RAMAN
# =========================================================
with st.sidebar:
    st.header("Parâmetros Raman")

    fit_model = st.selectbox(
        "Ajuste de pico",
        [None, "gauss", "lorentz", "voigt"],
    )

    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima", 1, 500, 5)

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Raman", "Questionário", "Otimizador Estatístico (PCA + Clusters)"]
)

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
        if sample_file is None:
            st.warning("Faça upload de um espectro.")
        else:
            st.session_state.raman_results = rp.process_raman_spectrum_with_groups(
                sample_file,
                peak_height=peak_height,
                peak_distance=peak_distance,
                peak_prominence=peak_prominence,
                fit_model=fit_model,
            )
            st.success("Espectro processado com sucesso.")

    if st.session_state.raman_results:
        data = st.session_state.raman_results

        st.subheader("Espectro processado")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"], lw=1.4)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade normalizada (u.a.)")
        st.pyplot(fig)

        st.subheader("Features Raman extraídas (ML-ready)")
        st.json(data["features"])

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Metadados dos Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])

    if q_file is not None:
        st.session_state.questionnaire_df = pd.read_csv(q_file)
        st.success("Questionário carregado com sucesso.")

    if st.session_state.questionnaire_df is not None:
        st.subheader("Pré-visualização do questionário")
        st.dataframe(
            st.session_state.questionnaire_df.head(),
            use_container_width=True,
        )

# =========================================================
# ABA 3 — OTIMIZADOR ESTATÍSTICO
# =========================================================
with tab3:
    st.header("Integração Raman × Questionário")

    if st.session_state.raman_results is None:
        st.info("Processe ao menos um espectro Raman primeiro.")
    elif st.session_state.questionnaire_df is None:
        st.info("Carregue o questionário para integração estatística.")
    else:
        # ---------------------------------------------
        # CONSTRUÇÃO DO DATASET REAL (AÇÃO EXPLÍCITA)
        # ---------------------------------------------
        st.subheader("Construção do dataset analítico")

        genero = st.selectbox("Gênero", ["F", "M"])
        fumante = st.selectbox("Fumante", ["não", "sim"])
        doenca = st.text_input("Doença declarada", value="controle")

        if st.button("➕ Adicionar amostra ao dataset"):
            features = st.session_state.raman_results["features"]

            row = {
                **features,
                "genero": genero,
                "fumante": fumante,
                "doenca": doenca,
            }

            st.session_state.ml_dataset = pd.concat(
                [st.session_state.ml_dataset, pd.DataFrame([row])],
                ignore_index=True,
            )

            st.success("Amostra adicionada ao dataset.")

        if st.session_state.ml_dataset.empty:
            st.info("Nenhuma amostra adicionada ainda.")
        else:
            df = st.session_state.ml_dataset.copy()

            st.subheader("Dataset consolidado")
            st.dataframe(df, use_container_width=True)

            # ---------------------------------------------
            # PREPARAÇÃO NUMÉRICA (PADRÃO ARTIGO)
            # ---------------------------------------------
            feature_cols = [
                c for c in df.columns
                if c not in ["genero", "fumante", "doenca"]
            ]

            X = df[feature_cols].fillna(0.0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ---------------------------------------------
            # PCA
            # ---------------------------------------------
            st.subheader("Análise de Componentes Principais (PCA)")

            n_components = st.slider(
                "Número de componentes principais",
                min_value=2,
                max_value=min(6, X.shape[1]),
                value=2,
            )

            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            st.write(
                "Variância explicada acumulada:",
                np.round(np.cumsum(pca.explained_variance_ratio_), 3),
            )

            # ---------------------------------------------
            # CLUSTERING
            # ---------------------------------------------
            st.subheader("Clustering não supervisionado (K-means)")

            k = st.slider("Número de clusters", 2, 6, 3)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_pca)

            df["cluster"] = clusters

            # ---------------------------------------------
            # PLOT PCA
            # ---------------------------------------------
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

            # ---------------------------------------------
            # ESTATÍSTICAS POR CLUSTER
            # ---------------------------------------------
            st.subheader("Distribuições estatísticas por cluster")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Gênero × Cluster")
                st.dataframe(pd.crosstab(df["cluster"], df["genero"]))

            with col2:
                st.write("Fumante × Cluster")
                st.dataframe(pd.crosstab(df["cluster"], df["fumante"]))

            with col3:
                st.write("Doença × Cluster")
                st.dataframe(pd.crosstab(df["cluster"], df["doenca"]))

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption("BioRaman • Análise Raman integrada • Uso científico • Marcela Veiga")
