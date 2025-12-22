# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma Integrada
Raman + Questionário + Estatística + Machine Learning
⚠ Uso exclusivo em pesquisa. NÃO é diagnóstico médico.
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
st.title("BioSensor — Plataforma Integrada")

# =========================================================
# SESSION STATE
# =========================================================
if "raman_results" not in st.session_state:
    st.session_state.raman_results = None

if "raman_table" not in st.session_state:
    st.session_state.raman_table = pd.DataFrame()

if "questionnaire" not in st.session_state:
    st.session_state.questionnaire = pd.DataFrame()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Parâmetros Raman")
    fit_model = st.selectbox(
        "Ajuste de picos",
        [None, "gauss", "lorentz", "voigt"],
        index=0,
    )
    peak_height = st.slider("Altura mínima", 0.0, 1.0, 0.03, 0.01)
    peak_prominence = st.slider("Proeminência", 0.0, 1.0, 0.03, 0.01)
    peak_distance = st.slider("Distância mínima", 1, 500, 5)

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["Raman", "Questionário / Pacientes", "Otimizador & Estatísticas"]
)

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    sample_code = st.text_input(
        "Código da amostra",
        value=f"AMOSTRA_{uuid.uuid4().hex[:6].upper()}",
    )

    uploaded = st.file_uploader(
        "Upload do espectro Raman",
        type=["txt", "csv", "xls", "xlsx"],
    )

    if uploaded and st.button("▶ Processar espectro"):
        res = rp.process_raman_spectrum_with_groups(
            uploaded,
            peak_height=peak_height,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
            fit_model=fit_model,
        )

        st.session_state.raman_results = res

        row = {
            "sample_code": sample_code,
            **res["features"],
        }

        st.session_state.raman_table = pd.concat(
            [st.session_state.raman_table, pd.DataFrame([row])],
            ignore_index=True,
        )

        st.success("Espectro processado e features armazenadas.")

    if st.session_state.raman_results:
        data = st.session_state.raman_results

        st.subheader("Espectro processado")
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

    q_file = st.file_uploader(
        "Upload CSV do questionário",
        type=["csv"],
    )

    if q_file:
        df_q = pd.read_csv(q_file)
        st.session_state.questionnaire = df_q
        st.success("Questionário carregado.")

    if not st.session_state.questionnaire.empty:
        st.subheader("Dados do questionário")
        st.dataframe(st.session_state.questionnaire, use_container_width=True)

# =========================================================
# ABA 3 — OTIMIZADOR & ESTATÍSTICA
# =========================================================
with tab3:
    st.header("Otimizador — Estatística & Machine Learning")

    if st.session_state.raman_table.empty or st.session_state.questionnaire.empty:
        st.info("É necessário ter Raman + Questionário carregados.")
    else:
        # ----------------------------------------------
        # INTEGRAÇÃO RAMAN × QUESTIONÁRIO
        # ----------------------------------------------
        df = pd.merge(
            st.session_state.raman_table,
            st.session_state.questionnaire,
            on="sample_code",
            how="inner",
        )

        st.subheader("Base integrada (Raman × Questionário)")
        st.dataframe(df, use_container_width=True)

        # ----------------------------------------------
        # ESTATÍSTICAS DESCRITIVAS
        # ----------------------------------------------
        st.subheader("Estatísticas populacionais")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Distribuição por gênero**")
            fig, ax = plt.subplots()
            df["genero"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("**Fumantes**")
            fig, ax = plt.subplots()
            df["fumante"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col3:
            st.markdown("**Doenças declaradas**")
            fig, ax = plt.subplots()
            df["doenca"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        # ----------------------------------------------
        # MACHINE LEARNING
        # ----------------------------------------------
        st.markdown("---")
        st.subheader("Machine Learning — Random Forest")

        label_col = st.selectbox(
            "Variável alvo (label)",
            ["doenca", "fumante", "genero"],
        )

        X = df.drop(columns=["sample_code", "genero", "fumante", "doenca"])
        y = df[label_col]

        ml_df = pd.concat([X, y.rename("label")], axis=1)

        if st.button("Treinar Random Forest"):
            result = train_random_forest_from_features(
                ml_df,
                label_col="label",
                config=MLConfig(),
            )

            st.metric("Acurácia", f"{result.accuracy:.2f}")
            st.text(result.report_text)

            st.subheader("Importância das features Raman")
            st.dataframe(result.feature_importances.head(20))

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
st.caption("BioRaman • Estatística Raman × Questionário • Marcela Veiga")
