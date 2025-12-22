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
    st.header("Otimizador — Estatística Raman × Questionário")

    if st.session_state.raman_results is None or st.session_state.questionnaire_df is None:
        st.info("Carregue Raman e Questionário para habilitar o otimizador.")
    else:
        if st.button("📊 Gerar estatísticas integradas"):
            st.session_state.stats_ready = True

        if st.session_state.stats_ready:
            df_q = st.session_state.questionnaire_df.copy()
            features = st.session_state.raman_results["features"]

            # ----------------------------
            # Estatísticas demográficas
            # ----------------------------
            st.subheader("Distribuição demográfica")

            for col in ["genero", "fumante", "doenca"]:
                if col in df_q.columns:
                    fig, ax = plt.subplots()
                    df_q[col].value_counts().plot(kind="bar", ax=ax)
                    ax.set_title(f"Distribuição por {col}")
                    st.pyplot(fig)

            # ----------------------------
            # Dataset ML (1 amostra exemplo)
            # ----------------------------
            st.subheader("Features Raman (exemplo)")
            df_feat = pd.DataFrame([features])
            st.dataframe(df_feat, use_container_width=True)

            # ----------------------------
            # ML (se houver labels)
            # ----------------------------
            if "doenca" in df_q.columns:
                label = df_q["doenca"].iloc[0]
                row = {**features, "label": label}
                st.session_state.ml_dataset = pd.DataFrame([row])

                if st.button("🚀 Treinar Random Forest (demo)"):
                    result = train_random_forest_from_features(
                        st.session_state.ml_dataset,
                        config=MLConfig(),
                    )
                    st.metric("Acurácia", f"{result.accuracy:.2f}")
                    st.text(result.report_text)
