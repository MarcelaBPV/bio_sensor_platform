# app.py
# -*- coding: utf-8 -*-

"""
BioRaman — Plataforma Integrada
Processamento Raman + Machine Learning + Estatística + Supabase

⚠ Uso exclusivo em pesquisa. NÃO é diagnóstico médico.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import uuid

import raman_processing as rp
from ml_otimizador import (
    train_random_forest_from_features,
    MLConfig,
)

from supabase_repository import (
    insert_sample,
    insert_spectrum,
    insert_peaks,
    insert_ml_features,
)

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(page_title="BioRaman", layout="wide")
st.title("BioSensor — Plataforma Integrada")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

# =========================================================
# SESSION STATE
# =========================================================
for key, default in {
    "raman_results": None,
    "ml_dataset": pd.DataFrame(),
    "questionario": None,
    "last_sample_id": None,
    "last_spectrum_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
tab1, tab2, tab3, tab4 = st.tabs(
    ["Raman", "Questionário / Pacientes", "Machine Learning", "Otimizador / Estatísticas"]
)

# =========================================================
# ABA 1 — RAMAN
# =========================================================
with tab1:
    st.header("Processamento Raman")

    sample_file = st.file_uploader(
        "Upload do espectro da amostra",
        type=["txt", "csv", "xls", "xlsx"],
    )

    substrate_file = None
    if use_substrate:
        substrate_file = st.file_uploader(
            "Upload do espectro do substrato",
            type=["txt", "csv", "xls", "xlsx"],
        )

    if sample_file and st.button("▶ Processar espectro"):
        st.session_state.raman_results = rp.process_raman_spectrum_with_groups(
            sample_file,
            substrate_file_like=substrate_file,
            peak_height=peak_height,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
            fit_model=fit_model,
        )
        st.success("Processamento concluído.")

    if st.session_state.raman_results:
        data = st.session_state.raman_results

        st.subheader("Espectro processado")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["x_proc"], data["y_proc"], lw=1.6)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensidade (u.a.)")
        st.pyplot(fig)

        if data["peaks"]:
            df_peaks = pd.DataFrame([
                {
                    "Raman shift (cm⁻¹)": p.position_cm1,
                    "Intensidade": p.intensity,
                    "Grupo molecular": p.group,
                    "FWHM": p.width,
                } for p in data["peaks"]
            ])
            st.subheader("Picos detectados")
            st.dataframe(df_peaks, use_container_width=True)

        st.markdown("---")
        st.subheader("Persistência (Supabase)")

        sample_code = st.text_input(
            "Código da amostra",
            value=f"AMOSTRA_{uuid.uuid4().hex[:6].upper()}",
        )

        sample_type = st.selectbox(
            "Tipo de amostra",
            ["sangue", "controle", "substrato", "outro"],
        )

        if st.button("💾 Salvar espectro no Supabase"):
            sample_id = insert_sample(sample_code, sample_type, {"origem": "BioRaman"})
            spectrum_id = insert_spectrum(
                sample_id,
                "processed",
                data["x_proc"].tolist(),
                data["y_proc"].tolist(),
                data["meta"],
            )
            insert_peaks(spectrum_id, data["peaks"])
            st.session_state.last_sample_id = sample_id
            st.session_state.last_spectrum_id = spectrum_id
            st.success("Espectro e picos salvos.")

# =========================================================
# ABA 2 — QUESTIONÁRIO
# =========================================================
with tab2:
    st.header("Questionário / Pacientes")

    q_file = st.file_uploader("Upload CSV do questionário", type=["csv"])
    if q_file:
        st.session_state.questionario = pd.read_csv(q_file)
        st.success("Questionário carregado.")

    if st.session_state.questionario is not None:
        st.dataframe(st.session_state.questionario.head(), use_container_width=True)

# =========================================================
# ABA 3 — MACHINE LEARNING (TREINAMENTO)
# =========================================================
with tab3:
    st.header("Machine Learning — Random Forest")

    if st.session_state.raman_results is None:
        st.info("Processe um espectro na Aba Raman.")
    else:
        label = st.text_input("Rótulo da amostra (classe)")

        if st.button("➕ Adicionar ao dataset ML"):
            row = {
                **st.session_state.raman_results["features"],
                "label": label,
            }
            st.session_state.ml_dataset = pd.concat(
                [st.session_state.ml_dataset, pd.DataFrame([row])],
                ignore_index=True,
            )
            st.success("Amostra adicionada.")

        if not st.session_state.ml_dataset.empty:
            st.dataframe(st.session_state.ml_dataset, use_container_width=True)

            if st.button("Treinar Random Forest"):
                result = train_random_forest_from_features(
                    st.session_state.ml_dataset,
                    label_col="label",
                    config=MLConfig(),
                )

                st.metric("Acurácia", f"{result.accuracy:.2f}")
                st.text(result.report_text)

                st.subheader("Importância das features")
                st.dataframe(result.feature_importances.head(15))

                fig, ax = plt.subplots(figsize=(6, 4))
                result.feature_importances.head(10).plot(
                    kind="barh", x="feature", y="importance", ax=ax
                )
                ax.invert_yaxis()
                st.pyplot(fig)

                if st.session_state.last_sample_id and st.session_state.last_spectrum_id:
                    if st.button("💾 Salvar features no Supabase"):
                        insert_ml_features(
                            st.session_state.last_sample_id,
                            st.session_state.last_spectrum_id,
                            st.session_state.raman_results["features"],
                            label,
                        )
                        st.success("Features salvas.")

# =========================================================
# ABA 4 — OTIMIZADOR / ESTATÍSTICAS
# =========================================================
with tab4:
    st.header("Otimizador — Estatísticas Exploratórias")

    df = st.session_state.questionario
    if df is None:
        st.info("Carregue um questionário na Aba 2.")
    else:
        st.subheader("Distribuição de participantes")

        for col, title in [
            ("genero", "Gênero"),
            ("fumante", "Fumante"),
            ("doenca", "Doenças declaradas"),
        ]:
            if col in df.columns:
                st.markdown(f"### {title}")
                counts = df[col].value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind="bar", ax=ax)
                ax.set_ylabel("Quantidade")
                st.pyplot(fig)

        st.subheader("Resumo estatístico")
        st.dataframe(df.describe(include="all").T)

# =========================================================
# RODAPÉ
# =========================================================
st.markdown("---")
st.caption("BioRaman • Plataforma científica • Marcela Veiga")
