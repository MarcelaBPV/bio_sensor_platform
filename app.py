# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma Integrada
Raman • Questionário • Otimizador ML
⚠ Uso exclusivo em pesquisa.
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
# SESSION STATE SEGURO
# =========================================================
if "raman_results" not in st.session_state:
    st.session_state.raman_results = None

if "ml_dataset" not in st.session_state:
    st.session_state.ml_dataset = pd.DataFrame()

if "ml_result" not in st.session_state:
    st.session_state.ml_result = None

if "sample_code" not in st.session_state:
    st.session_state.sample_code = f"AMOSTRA_{uuid.uuid4().hex[:6].upper()}"

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "🔬 Raman",
    "📋 Questionário",
    "📊 Otimizador / Estatísticas"
])

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    with st.form("raman_form"):
        sample_file = st.file_uploader(
            "Upload espectro Raman",
            type=["txt", "csv", "xls", "xlsx"]
        )

        fit_model = st.selectbox(
            "Ajuste de pico",
            [None, "gauss", "lorentz", "voigt"]
        )

        submit_raman = st.form_submit_button("▶ Processar")

    if submit_raman and sample_file:
        st.session_state.raman_results = rp.process_raman_spectrum_with_groups(
            sample_file,
            fit_model=fit_model
        )

    if st.session_state.raman_results:
        data = st.session_state.raman_results

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"])
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

        st.subheader("Picos detectados")
        df_peaks = pd.DataFrame([
            {
                "cm⁻¹": p.position_cm1,
                "Intensidade": p.intensity,
                "Grupo": p.group
            }
            for p in data["peaks"]
        ])
        st.dataframe(df_peaks, use_container_width=True)

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Dados Clínicos")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])

    if q_file:
        df_q = pd.read_csv(q_file)
        st.session_state.questionnaire = df_q
        st.dataframe(df_q.head(), use_container_width=True)

# =========================================================
# ABA 3 — OTIMIZADOR / ESTATÍSTICAS
# =========================================================
with tab3:
    st.header("Otimizador ML & Estatísticas")

    if st.session_state.raman_results is None:
        st.info("Processe um espectro Raman primeiro.")
        st.stop()

    label = st.text_input("Classe da amostra (ex.: controle, diabetes)")

    if st.button("➕ Adicionar ao dataset ML"):
        row = {
            **st.session_state.raman_results["features"],
            "label": label,
        }
        st.session_state.ml_dataset = pd.concat(
            [st.session_state.ml_dataset, pd.DataFrame([row])],
            ignore_index=True,
        )

    if not st.session_state.ml_dataset.empty:
        st.subheader("Dataset ML")
        st.dataframe(st.session_state.ml_dataset)

        if st.button("🚀 Treinar Random Forest"):
            st.session_state.ml_result = train_random_forest_from_features(
                st.session_state.ml_dataset,
                label_col="label",
                config=MLConfig(),
            )

    if st.session_state.ml_result:
        res = st.session_state.ml_result

        st.metric("Acurácia", f"{res.accuracy:.2f}")
        st.text(res.report_text)

        st.subheader("Importância das features")
        st.dataframe(res.feature_importances.head(10))

    # ---------- ESTATÍSTICAS DO QUESTIONÁRIO ----------
    if "questionnaire" in st.session_state:
        st.markdown("---")
        st.subheader("Estatísticas Questionário")

        df_q = st.session_state.questionnaire

        for col in ["genero", "fumante", "doenca"]:
            if col in df_q.columns:
                fig, ax = plt.subplots()
                df_q[col].value_counts(normalize=True).plot(
                    kind="bar",
                    ax=ax
                )
                ax.set_title(f"Distribuição por {col}")
                st.pyplot(fig)
