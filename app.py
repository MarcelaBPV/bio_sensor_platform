# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma integrada
Processamento Raman + Machine Learning
⚠ Uso em pesquisa. NÃO é diagnóstico médico.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

import raman_processing as rp
from ml_otimizador import (
    train_random_forest_from_features,
    MLConfig,
)

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Plataforma Integrada")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

# =========================================================
# SESSION STATE
# =========================================================
if "raman_results" not in st.session_state:
    st.session_state.raman_results = None

if "ml_dataset" not in st.session_state:
    st.session_state.ml_dataset = pd.DataFrame()

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Parâmetros Raman")

    use_substrate = st.checkbox("Subtrair substrato", False)
    fit_model = st.selectbox(
        "Ajuste de picos",
        [None, "gauss", "lorentz", "voigt"],
        index=0,
    )

    st.markdown("---")
    st.subheader("Detecção de picos")
    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima", 1, 500, 5)

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Raman", "Questionário / Pacientes", "Machine Learning"]
)

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    sample_file = st.file_uploader(
        "Upload espectro da amostra",
        type=["txt", "csv", "xls", "xlsx"],
        key="sample",
    )

    substrate_file = None
    if use_substrate:
        substrate_file = st.file_uploader(
            "Upload espectro do substrato",
            type=["txt", "csv", "xls", "xlsx"],
            key="substrate",
        )

    if sample_file and st.button("▶ Processar espectro"):
        res = rp.process_raman_spectrum_with_groups(
            sample_file,
            substrate_file_like=substrate_file,
            peak_height=peak_height,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
            fit_model=fit_model,
        )
        st.session_state.raman_results = res
        st.success("Processamento concluído.")

    # ---------------- VISUALIZAÇÃO ----------------
    if st.session_state.raman_results:
        data = st.session_state.raman_results

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"], lw=1.6)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

        # Tabela de picos
        peaks = data["peaks"]
        if peaks:
            df_peaks = pd.DataFrame(
                [{
                    "Raman shift": p.position_cm1,
                    "Intensidade": p.intensity,
                    "Grupo molecular": p.group,
                    "FWHM": p.width,
                } for p in peaks]
            )
            st.subheader("Picos detectados")
            st.dataframe(df_peaks, use_container_width=True)

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])
    if q_file:
        df_q = pd.read_csv(q_file)
        st.dataframe(df_q.head(), use_container_width=True)

# =========================================================
# ABA 3 — MACHINE LEARNING
# =========================================================
with tab3:
    st.header("Machine Learning — Random Forest")

    if st.session_state.raman_results is None:
        st.info("Processe um espectro na Aba Raman primeiro.")
    else:
        label = st.text_input("Rótulo da amostra (ex.: controle, diabetes, asma)")

        if st.button("➕ Adicionar amostra ao dataset ML"):
            features = st.session_state.raman_results["features"]
            row = {**features, "label": label}
            st.session_state.ml_dataset = pd.concat(
                [st.session_state.ml_dataset, pd.DataFrame([row])],
                ignore_index=True,
            )
            st.success("Amostra adicionada ao dataset.")

        if not st.session_state.ml_dataset.empty:
            st.subheader("Dataset ML")
            st.dataframe(st.session_state.ml_dataset, use_container_width=True)

            if st.button("🚀 Treinar Random Forest"):
                result = train_random_forest_from_features(
                    st.session_state.ml_dataset,
                    label_col="label",
                    config=MLConfig(),
                )

                st.subheader("Desempenho do modelo")
                st.metric("Acurácia", f"{result.accuracy:.2f}")
                st.text(result.report_text)

                st.subheader("Importância das features")
                st.dataframe(
                    result.feature_importances.head(15),
                    use_container_width=True,
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                result.feature_importances.head(10).plot(
                    kind="barh",
                    x="feature",
                    y="importance",
                    ax=ax,
                )
                ax.invert_yaxis()
                st.pyplot(fig)

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption("BioSensor • Uso em pesquisa - Macela Veiga")
