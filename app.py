# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import uuid

import raman_processing as rp
from ml_otimizador import train_random_forest_from_features, MLConfig
from supabase_repository import (
    insert_sample, insert_spectrum, insert_peaks, insert_ml_features
)

st.set_page_config(page_title="BioRaman", layout="wide")
st.title("🧬 BioRaman — Plataforma Integrada")

# =========================================================
# SESSION STATE
# =========================================================
for k in [
    "raman_results",
    "ml_dataset",
    "questionario_df",
    "last_sample_id",
    "last_spectrum_id",
]:
    if k not in st.session_state:
        st.session_state[k] = None if "df" not in k else pd.DataFrame()

# =========================================================
# ABAS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Raman",
    "Questionário",
    "Estatística Raman × Questionário",
    "Otimizador / ML"
])

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    sample_file = st.file_uploader("Upload do espectro Raman")

    if sample_file and st.button("▶ Processar espectro"):
        with st.spinner("Processando espectro..."):
            st.session_state.raman_results = rp.process_raman_spectrum_with_groups(sample_file)
        st.success("Processamento concluído.")

    if st.session_state.raman_results:
        data = st.session_state.raman_results

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"])
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

        if st.button("💾 Salvar no Supabase"):
            with st.spinner("Salvando no banco..."):
                sid = insert_sample(
                    sample_code=f"AMOSTRA_{uuid.uuid4().hex[:6]}",
                    sample_type="sangue",
                    metadata={}
                )
                spid = insert_spectrum(
                    sid, "processed",
                    data["x_proc"].tolist(),
                    data["y_proc"].tolist(),
                    data["meta"]
                )
                insert_peaks(spid, data["peaks"])

                st.session_state.last_sample_id = sid
                st.session_state.last_spectrum_id = spid

            st.success("Salvo com sucesso.")

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])
    if q_file:
        st.session_state.questionario_df = pd.read_csv(q_file)
        st.dataframe(st.session_state.questionario_df.head())

# =========================================================
# ABA 3 — ESTATÍSTICA RAMAN × QUESTIONÁRIO
# =========================================================
with tab3:
    st.header("Estatística Integrada")

    df_q = st.session_state.questionario_df
    data = st.session_state.raman_results

    if df_q is None or df_q.empty or data is None:
        st.info("Carregue espectro Raman e questionário.")
    else:
        st.subheader("Distribuição por gênero")
        if "genero" in df_q.columns:
            st.bar_chart(df_q["genero"].value_counts())

        st.subheader("Fumantes vs Não fumantes")
        if "fumante" in df_q.columns:
            st.bar_chart(df_q["fumante"].value_counts())

        st.subheader("Doenças declaradas")
        if "doenca" in df_q.columns:
            st.bar_chart(df_q["doenca"].value_counts())

# =========================================================
# ABA 4 — OTIMIZADOR / ML
# =========================================================
with tab4:
    st.header("Otimizador — Machine Learning")

    if st.session_state.raman_results is None:
        st.info("Processe espectros primeiro.")
    else:
        label = st.text_input("Classe / rótulo")

        if st.button("➕ Adicionar ao dataset ML"):
            row = {
                **st.session_state.raman_results["features"],
                "label": label
            }
            st.session_state.ml_dataset = pd.concat(
                [st.session_state.ml_dataset, pd.DataFrame([row])],
                ignore_index=True
            )
            st.success("Amostra adicionada.")

        if st.session_state.ml_dataset is not None and not st.session_state.ml_dataset.empty:
            st.dataframe(st.session_state.ml_dataset)

            if st.button("🚀 Treinar Random Forest"):
                result = train_random_forest_from_features(
                    st.session_state.ml_dataset,
                    label_col="label",
                    config=MLConfig()
                )

                st.metric("Acurácia", f"{result.accuracy:.2f}")
                st.text(result.report_text)
                st.dataframe(result.feature_importances.head(10))
